"""
Overnight curriculum training for HeXO AlphaZero.

PARALLEL: persistent inference server + CPU workers per phase.
Workers run continuously, main thread collects games and trains.

Usage:
    python overnight_train.py
"""

import os
import sys
import time
import copy
import random
import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOG_PATH = None
HEARTBEAT_PATH = None

def log(msg=""):
    print(msg, flush=True)
    if LOG_PATH is not None:
        fd = os.open(LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        os.write(fd, (msg + "\n").encode('utf-8'))
        os.close(fd)

def heartbeat(status="alive"):
    """Write timestamp to heartbeat file so external monitor can detect hangs."""
    if HEARTBEAT_PATH is not None:
        fd = os.open(HEARTBEAT_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.write(fd, f"{time.time():.0f} {status}\n".encode('utf-8'))
        os.close(fd)

from hexo.game import HeXOGame
from alphazero.model import ResNet, collate_states, uncollate_policy
from alphazero.mcts import MCTS
from alphazero.train import AlphaZero
from alphazero.self_play import self_play_worker
from alphazero.inference import InferenceServer


# ============================================================================
# Curriculum phases
# ============================================================================
PHASES = [
    # Phase 1 already done
    {
        'name': 'Phase 1: win=3, dist=2',
        'win_length': 3, 'max_placement_dist': 2,
        'duration_minutes': 60,
        'num_searches': 25, 'max_moves_per_game': 30,
        'num_workers': 5,
        'collect_seconds': 30,
        'seed_random_games': 1000, 'seed_epochs': 8,
        'temperature': 1.5, 'dirichlet_epsilon': 0.5, 'dirichlet_alpha': 0.5,
        'lr': 0.001,
    },
    # 3-hour focused run: 4 workers (less VRAM), fewer warmup games (faster)
    {
        'name': 'Phase 2: win=3, dist=4',
        'win_length': 3, 'max_placement_dist': 4,
        'duration_minutes': 30,
        'num_searches': 30, 'max_moves_per_game': 50,
        'num_workers': 5,
        'collect_seconds': 30,
        'seed_random_games': 2000, 'seed_epochs': 6,
        'warmup_nn_games': 2000, 'warmup_epochs': 6, 'warmup_temp': 2.0,
        'temperature': 1.5, 'dirichlet_epsilon': 0.5, 'dirichlet_alpha': 0.5,
        'lr': 0.001,
    },
    {
        'name': 'Phase 3: win=4, dist=3',
        'win_length': 4, 'max_placement_dist': 3,
        'duration_minutes': 40,
        'num_searches': 50, 'max_moves_per_game': 80,
        'num_workers': 5,
        'collect_seconds': 45,
        'seed_random_games': 5000, 'seed_epochs': 6,
        'warmup_nn_games': 2000, 'warmup_epochs': 6, 'warmup_temp': 2.0,
        'temperature': 1.5, 'dirichlet_epsilon': 0.5, 'dirichlet_alpha': 0.5,
        'lr': 0.001,
    },
    {
        'name': 'Phase 4: win=5, dist=3',
        'win_length': 5, 'max_placement_dist': 3,
        'duration_minutes': 50,
        'num_searches': 8, 'max_moves_per_game': 80,
        'num_workers': 8,
        'collect_seconds': 30,
        'seed_random_games': 3000, 'seed_epochs': 4,
        'warmup_nn_games': 1000, 'warmup_epochs': 4, 'warmup_temp': 2.0,
        'temperature': 1.5, 'dirichlet_epsilon': 0.5, 'dirichlet_alpha': 0.5,
        'lr': 0.0005,
    },
    {
        'name': 'Phase 5: win=5, dist=5',
        'win_length': 5, 'max_placement_dist': 5,
        'duration_minutes': 50,
        'num_searches': 8, 'max_moves_per_game': 100,
        'num_workers': 8,
        'collect_seconds': 30,
        'seed_random_games': 2000, 'seed_epochs': 4,
        'warmup_nn_games': 500, 'warmup_epochs': 4, 'warmup_temp': 2.0,
        'temperature': 1.5, 'dirichlet_epsilon': 0.4, 'dirichlet_alpha': 0.4,
        'lr': 0.0003,
    },
    {
        'name': 'Phase 6: win=5, dist=8 (full rules)',
        'win_length': 5, 'max_placement_dist': 8,
        'duration_minutes': 50,
        'num_searches': 8, 'max_moves_per_game': 120,
        'num_workers': 8,
        'collect_seconds': 30,
        'seed_random_games': 2000, 'seed_epochs': 4,
        'warmup_nn_games': 500, 'warmup_epochs': 4, 'warmup_temp': 2.0,
        'temperature': 1.5, 'dirichlet_epsilon': 0.35, 'dirichlet_alpha': 0.3,
        'lr': 0.0003,
    },
]


def generate_random_games(game, num_games, max_moves):
    """Random play data — only winning games kept."""
    all_samples = []
    wins = 0
    for _ in range(num_games):
        state = game.get_initial_state()
        memory = []
        for move_num in range(max_moves):
            valid = game.get_valid_moves(state)
            legal = np.where(valid > 0)[0]
            if len(legal) == 0:
                break
            action_probs = np.zeros(len(valid), dtype=np.float32)
            action_probs[legal] = 1.0 / len(legal)
            encoded = game.get_encoded_state(state)
            player = state.current_player
            memory.append((encoded, action_probs, player))
            action = np.random.choice(legal)
            state = game.get_next_state(state, action, state.current_player)
            if state.winner != 0:
                wins += 1
                last_player = state.move_history[-1][2]
                for enc, pi, pl in memory:
                    outcome = 1.0 if pl == last_player else -1.0
                    all_samples.append((enc, pi, outcome))
                break
    return all_samples, wins, num_games


@torch.no_grad()
def nn_fast_selfplay(model, game, num_games, max_moves, temperature=2.0, device='cpu'):
    """Fast self-play using NN policy directly (no MCTS). Only keeps wins."""
    all_samples = []
    wins = 0
    model.eval()

    for _ in range(num_games):
        state = game.get_initial_state()
        memory = []
        for move_num in range(max_moves):
            valid = game.get_valid_moves(state)
            legal = np.where(valid > 0)[0]
            if len(legal) == 0:
                break

            # NN inference for policy
            encoded = game.get_encoded_state(state)
            state_t = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(device)
            policy_logits, value = model(state_t)
            policy_logits = policy_logits.squeeze(0).cpu().numpy()

            # Mask invalid moves and apply temperature
            action_size = len(valid)
            logits = policy_logits[:action_size]
            logits[valid == 0] = -1e9
            logits = logits / max(temperature, 0.01)
            exp_logits = np.exp(logits - logits.max())
            probs = exp_logits / exp_logits.sum()

            player = state.current_player
            memory.append((encoded, probs, player))

            action = np.random.choice(action_size, p=probs)
            state = game.get_next_state(state, action, state.current_player)

            if state.winner != 0:
                wins += 1
                last_player = state.move_history[-1][2]
                for enc, pi, pl in memory:
                    outcome = 1.0 if pl == last_player else -1.0
                    all_samples.append((enc, pi, outcome))
                break

    return all_samples, wins, num_games


def inference_server_process(model_state_dict, model_config, device_str,
                             request_queue, response_queues, stop_event,
                             model_update_queue, batch_size=32):
    """Inference server that can hot-reload model weights."""
    device = torch.device(device_str)
    game = HeXOGame()
    model = ResNet(game, **model_config, device=device)
    model.load_state_dict(model_state_dict)
    model.eval()

    server = InferenceServer(model, device, batch_size=batch_size)

    import queue as queue_module
    pending = []

    while not stop_event.is_set():
        # Check for model update
        try:
            new_state = model_update_queue.get_nowait()
            model.load_state_dict(new_state)
            model.eval()
        except:
            pass

        # Collect a request
        try:
            req = request_queue.get(timeout=0.01)
            pending.append(req)
        except:
            pass

        # Process batch
        if pending and (len(pending) >= batch_size or len(pending) > 0):
            results = server.process_batch(pending)
            for wid, policy, value in results:
                if wid in response_queues:
                    response_queues[wid].put((policy, value))
            pending = []

    # Flush
    if pending:
        results = server.process_batch(pending)
        for wid, policy, value in results:
            if wid in response_queues:
                response_queues[wid].put((policy, value))


def continuous_worker(worker_id, game_args, mcts_args, request_queue,
                      response_queue, result_queue, stop_event):
    """Worker that plays games continuously until stopped."""
    game = HeXOGame(max_placement_dist=game_args.get('max_placement_dist', 8),
                    win_length=game_args.get('win_length', 5))

    while not stop_event.is_set():
        self_play_worker(worker_id, game_args, mcts_args, request_queue,
                         response_queue, result_queue, 1, stop_event)


def run_phase(phase, model, device, save_dir, global_start):
    """Run one phase with persistent parallel workers."""
    name = phase['name']
    win_length = phase['win_length']
    dist = phase['max_placement_dist']
    duration_s = phase['duration_minutes'] * 60
    num_workers = phase.get('num_workers', 6)
    collect_s = phase.get('collect_seconds', 30)
    phase_start = time.time()

    log(f"\n{'#'*70}")
    log(f"# {name}")
    log(f"# win={win_length}, dist={dist}, sims={phase['num_searches']}, "
        f"workers={num_workers}, collect={collect_s}s")
    elapsed_total = (time.time() - global_start) / 60
    log(f"# Total elapsed: {elapsed_total:.0f}min")
    log(f"{'#'*70}\n")

    game = HeXOGame(max_placement_dist=dist, win_length=win_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=phase['lr'], weight_decay=1e-4)

    args = {
        'C': 2,
        'num_searches': phase['num_searches'],
        'dirichlet_epsilon': phase['dirichlet_epsilon'],
        'dirichlet_alpha': phase['dirichlet_alpha'],
        'temperature': phase['temperature'],
        'max_moves_per_game': phase['max_moves_per_game'],
        'batch_size': 64,
    }
    trainer = AlphaZero(model, optimizer, game, args)

    # --- Seed ---
    seed_data, seed_wins = [], 0
    num_seed = phase.get('seed_random_games', 0)
    if num_seed > 0:
        log(f"Seeding: {num_seed} random games...")
        t0 = time.time()
        seed_data, seed_wins, seed_total = generate_random_games(
            game, num_seed, phase['max_moves_per_game'])
        log(f"  {len(seed_data)} samples, {seed_wins}/{seed_total} wins "
            f"({100*seed_wins/seed_total:.0f}%), {time.time()-t0:.1f}s")
        if seed_wins > 0:
            # Filter out oversized states — scale limit with dist
            max_dim = 20 + dist * 6
            seed_data = [s for s in seed_data if s[0].shape[1] <= max_dim and s[0].shape[2] <= max_dim]
            log(f"  After size filter (max {max_dim}): {len(seed_data)} samples")
            # Cap pre-training data to keep it fast; full seed stays in replay buffer
            max_pretrain = 5000
            pretrain_data = seed_data if len(seed_data) <= max_pretrain else random.sample(seed_data, max_pretrain)
            epochs = min(phase['seed_epochs'], 4)
            log(f"Pre-training ({epochs} epochs on {len(pretrain_data)} samples)...")
            model.train()
            for epoch in range(epochs):
                losses = trainer.train(pretrain_data)
                if epoch == 0 or epoch == epochs - 1:
                    log(f"  Epoch {epoch}: p={losses['policy_loss']:.4f} v={losses['value_loss']:.4f}")

    # --- NN warmup: fast self-play with NN policy (no MCTS) to bootstrap ---
    warmup_games = phase.get('warmup_nn_games', 0)
    if warmup_games > 0:
        warmup_temp = phase.get('warmup_temp', 2.0)
        warmup_epochs = min(phase.get('warmup_epochs', 6), 4)
        log(f"NN warmup: {warmup_games} fast games (temp={warmup_temp})...")
        t0 = time.time()
        nn_data, nn_wins, nn_total = nn_fast_selfplay(
            model, game, warmup_games, phase['max_moves_per_game'],
            temperature=warmup_temp, device=device)
        log(f"  {len(nn_data)} samples, {nn_wins}/{nn_total} wins "
            f"({100*nn_wins/nn_total:.0f}%), {time.time()-t0:.1f}s")
        if nn_wins > 0:
            # Filter oversized + merge into seed data
            nn_data = [s for s in nn_data if s[0].shape[1] <= max_dim and s[0].shape[2] <= max_dim]
            seed_data.extend(nn_data)
            seed_wins += nn_wins
            max_pretrain = 5000
            pretrain_data = nn_data if len(nn_data) <= max_pretrain else random.sample(nn_data, max_pretrain)
            log(f"  Training on NN warmup ({warmup_epochs} epochs, {len(pretrain_data)} samples)...")
            model.train()
            for epoch in range(warmup_epochs):
                losses = trainer.train(pretrain_data)
                if epoch == 0 or epoch == warmup_epochs - 1:
                    log(f"  Epoch {epoch}: p={losses['policy_loss']:.4f} v={losses['value_loss']:.4f}")

    # Free VRAM from pre-training before spawning inference server
    torch.cuda.empty_cache()

    # --- Launch persistent workers ---
    model.eval()
    model_state = model.state_dict()
    model_config = {'num_resBlocks': len(model.backBone), 'num_hidden': model.num_hidden}

    request_queue = mp.Queue()
    response_queues = {wid: mp.Queue() for wid in range(num_workers)}
    result_queue = mp.Queue()
    stop_event = mp.Event()
    model_update_queue = mp.Queue()

    game_args = {'max_placement_dist': dist, 'win_length': win_length}
    mcts_args = {
        'C': args['C'], 'num_searches': args['num_searches'],
        'dirichlet_epsilon': args['dirichlet_epsilon'],
        'dirichlet_alpha': args['dirichlet_alpha'],
        'temperature': args['temperature'],
        'max_moves_per_game': args['max_moves_per_game'],
    }

    # Start inference server
    server_proc = mp.Process(
        target=inference_server_process,
        args=(model_state, model_config, str(device),
              request_queue, response_queues, stop_event, model_update_queue),
        kwargs={'batch_size': min(num_workers * 8, 64)},
    )
    server_proc.start()
    log(f"  Inference server started (PID {server_proc.pid})")

    # Start workers
    workers = []
    for wid in range(num_workers):
        p = mp.Process(
            target=continuous_worker,
            args=(wid, game_args, mcts_args, request_queue,
                  response_queues[wid], result_queue, stop_event),
        )
        p.start()
        workers.append(p)
    log(f"  {num_workers} workers started")

    # --- Main loop: collect games, train, push weights ---
    # Keep seed data separate so it never gets rotated out
    seed_samples = list(seed_data) if seed_wins > 0 else []
    mcts_buffer = []
    max_mcts_buffer = 40000
    iteration = 0
    total_wins = 0
    total_games = 0
    consecutive_no_wins = 0
    import queue as queue_module

    while (time.time() - phase_start) < duration_s:
        iter_start = time.time()
        heartbeat(f"phase={name} iter={iteration}")

        # Check worker health — restart dead workers
        for i, w in enumerate(workers):
            if not w.is_alive():
                log(f"  WARNING: worker {i} died, restarting")
                w = mp.Process(
                    target=continuous_worker,
                    args=(i, game_args, mcts_args, request_queue,
                          response_queues[i], result_queue, stop_event),
                )
                w.start()
                workers[i] = w
        if not server_proc.is_alive():
            log(f"  WARNING: inference server died, restarting")
            model_state = model.state_dict()
            server_proc = mp.Process(
                target=inference_server_process,
                args=(model_state, model_config, str(device),
                      request_queue, response_queues, stop_event, model_update_queue),
                kwargs={'batch_size': min(num_workers * 8, 64)},
            )
            server_proc.start()

        # Collect games for collect_s seconds
        new_samples = []
        wins = 0
        games_collected = 0
        collect_deadline = time.time() + collect_s

        while time.time() < collect_deadline and (time.time() - phase_start) < duration_s:
            try:
                game_samples = result_queue.get(timeout=1.0)
                new_samples.extend(game_samples)
                if any(abs(s[2]) > 0.5 for s in game_samples):
                    wins += 1
                games_collected += 1
            except queue_module.Empty:
                pass

        if not new_samples:
            time.sleep(5)
            continue

        total_wins += wins
        total_games += games_collected

        # Track consecutive iterations with no wins
        if wins == 0:
            consecutive_no_wins += 1
        else:
            consecutive_no_wins = 0

        # If no wins for 10+ iters, inject fresh NN self-play games
        if consecutive_no_wins > 0 and consecutive_no_wins % 10 == 0:
            log(f"  >> Injecting 500 NN games to refresh win signal...")
            nn_extra, nn_w, nn_t = nn_fast_selfplay(
                model, game, 500, phase['max_moves_per_game'],
                temperature=2.0, device=device)
            if nn_w > 0:
                nn_extra = [s for s in nn_extra if s[0].shape[1] <= max_dim and s[0].shape[2] <= max_dim]
                seed_samples.extend(nn_extra)
                log(f"     {nn_w}/{nn_t} wins, {len(nn_extra)} samples added to seed buffer")

        # MCTS replay buffer (seed data kept separate and always included)
        mcts_buffer.extend(new_samples)
        if len(mcts_buffer) > max_mcts_buffer:
            mcts_buffer = mcts_buffer[-max_mcts_buffer:]

        # Build training data: new samples + replay from both buffers
        # Always include seed data (win signal) to prevent value head collapse
        train_data = list(new_samples)
        if seed_samples:
            seed_batch = min(len(new_samples), len(seed_samples))
            train_data.extend(random.sample(seed_samples, seed_batch))
        if len(mcts_buffer) > len(new_samples):
            extra = min(len(new_samples), len(mcts_buffer))
            train_data.extend(random.sample(mcts_buffer, extra))

        # Train
        model.train()
        for epoch in range(4):
            losses = trainer.train(train_data)

        # Push updated weights to inference server
        model.eval()
        try:
            model_update_queue.put(model.state_dict())
        except:
            pass

        iter_time = time.time() - iter_start
        phase_elapsed = (time.time() - phase_start) / 60
        total_elapsed = (time.time() - global_start) / 60

        warn = " !! NO WINS" if consecutive_no_wins >= 5 else ""
        log(f"  Iter {iteration:3d} [{phase_elapsed:5.1f}m / {total_elapsed:5.1f}m] "
            f"{wins}/{games_collected}g ({len(new_samples)}+{len(train_data)-len(new_samples)} samp) "
            f"p={losses['policy_loss']:.4f} v={losses['value_loss']:.4f} "
            f"({iter_time:.0f}s) [buf={len(mcts_buffer)}+{len(seed_samples)}s]{warn}")

        if iteration % 5 == 0:
            phase_tag = name.split(':')[0].strip().replace(' ', '').lower()
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"{phase_tag}_iter{iteration}.pt"))
        iteration += 1

    # --- Shutdown workers ---
    stop_event.set()
    for w in workers:
        w.join(timeout=10)
        if w.is_alive():
            w.terminate()
    server_proc.join(timeout=10)
    if server_proc.is_alive():
        server_proc.terminate()

    # Drain remaining results
    remaining = 0
    while True:
        try:
            result_queue.get_nowait()
            remaining += 1
        except:
            break

    phase_tag = name.split(':')[0].strip().replace(' ', '').lower()
    torch.save(model.state_dict(), os.path.join(save_dir, f"{phase_tag}_final.pt"))

    phase_time = (time.time() - phase_start) / 60
    wr = 100 * total_wins / max(total_games, 1)
    log(f"\n  Phase done: {iteration} iters, {total_games} games, "
        f"{total_wins} wins ({wr:.0f}%), {phase_time:.1f}min")

    return model


def main():
    global LOG_PATH, HEARTBEAT_PATH
    base_dir = os.path.dirname(os.path.abspath(__file__))
    LOG_PATH = os.path.join(base_dir, 'overnight.log')
    HEARTBEAT_PATH = os.path.join(base_dir, 'heartbeat.txt')
    # Append to log if resuming
    resume_phase = int(os.environ.get('RESUME_PHASE', '1'))
    if resume_phase > 1:
        fd = os.open(LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        os.write(fd, b"\n=== RESUMING ===\n")
        os.close(fd)
    else:
        with open(LOG_PATH, 'w') as f:
            f.write("")
    heartbeat("starting")

    mp.set_start_method('spawn', force=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    log(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    save_dir = 'checkpoints_overnight'
    os.makedirs(save_dir, exist_ok=True)

    model = ResNet(HeXOGame(), num_resBlocks=10, num_hidden=128, device=device)

    # Resume from checkpoint if specified
    checkpoint_path = os.environ.get('RESUME_CHECKPOINT', '')
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        log(f"Loaded checkpoint: {checkpoint_path}")

    phases_to_run = PHASES[resume_phase - 1:]
    total_planned = sum(p['duration_minutes'] for p in phases_to_run)
    log(f"Phases: {len(phases_to_run)} (starting from {resume_phase}), "
        f"duration: {total_planned}min ({total_planned/60:.1f}h)")
    log(f"Model: 3.0M params | Parallel: persistent workers + batched GPU")

    global_start = time.time()
    for phase in phases_to_run:
        model = run_phase(phase, model, device, save_dir, global_start)

    torch.save(model.state_dict(), os.path.join(save_dir, 'model_final.pt'))
    total_time = (time.time() - global_start) / 60
    log(f"\n{'='*60}")
    log(f"DONE: {total_time:.0f}min ({total_time/60:.1f}h)")
    log(f"{'='*60}")


if __name__ == '__main__':
    main()
