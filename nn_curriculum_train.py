"""
NN-only curriculum training for HeXO AlphaZero.

APPROACH: Drop MCTS from training. Play many games simultaneously using the
NN's policy (batched GPU inference). Only keep winning games. Train on winners.
Final phase adds MCTS refinement once the policy is strong.

This is dramatically faster than MCTS self-play because:
- 1 forward pass per move (vs N simulations per move)
- All games batched into a single GPU call
- No multiprocessing queues, no deadlocks

At PLAY TIME, the trained model is paired with MCTS for maximum strength.

Usage:
    python nn_curriculum_train.py
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hexo.game import HeXOGame
from alphazero.model import ResNet, collate_states
from alphazero.train import AlphaZero
from alphazero.mcts import MCTS

LOG_PATH = None
HEARTBEAT_PATH = None
LOW_IMPACT = False  # Set via env var LOW_IMPACT=1 — reduces GPU/CPU usage for gaming

def log(msg=""):
    print(msg, flush=True)
    if LOG_PATH is not None:
        fd = os.open(LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        os.write(fd, (msg + "\n").encode('utf-8'))
        os.close(fd)

def heartbeat(status="alive"):
    if HEARTBEAT_PATH is not None:
        fd = os.open(HEARTBEAT_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.write(fd, f"{time.time():.0f} {status}\n".encode('utf-8'))
        os.close(fd)


# ============================================================================
# Curriculum
# ============================================================================
PHASES = [
    # NN-only phases: train policy through self-play
    # Phases graduate when win rate >= graduate_pct over last 5 iters (min time required)
    # Hard cap at max_minutes to avoid getting stuck forever
    # 7-HOUR OVERNIGHT CONFIG: generous time, low thresholds
    {
        'name': 'Phase A: win=4, dist=3',
        'win_length': 4, 'max_placement_dist': 3,
        'min_minutes': 15, 'max_minutes': 30,
        'graduate_pct': 10,
        'num_games': 200, 'num_play_workers': 4, 'max_moves': 80,
        'temperature': 1.5, 'lr': 0.001,
        'max_dim': 40,
        'seed_random_games': 3000,
    },
    {
        'name': 'Phase B: win=4, dist=5',
        'win_length': 4, 'max_placement_dist': 5,
        'min_minutes': 15, 'max_minutes': 30,
        'graduate_pct': 5,
        'num_games': 200, 'num_play_workers': 4, 'max_moves': 80,
        'temperature': 1.5, 'lr': 0.001,
        'max_dim': 55,
        'seed_random_games': 3000,
    },
    {
        'name': 'Phase C: win=5, dist=3',
        'win_length': 5, 'max_placement_dist': 3,
        'min_minutes': 15, 'max_minutes': 40,
        'graduate_pct': 5,
        'num_games': 200, 'num_play_workers': 4, 'max_moves': 80,
        'temperature': 1.5, 'lr': 0.001,
        'max_dim': 40,
        'seed_random_games': 5000,
    },
    {
        'name': 'Phase D: win=5, dist=4',
        'win_length': 5, 'max_placement_dist': 4,
        'min_minutes': 30, 'max_minutes': 90,
        'graduate_pct': 3,
        'num_games': 200, 'num_play_workers': 4, 'max_moves': 90,
        'temperature': 1.5, 'lr': 0.001,
        'max_dim': 45,
        'seed_random_games': 5000,
    },
    {
        'name': 'Phase E: win=5, dist=5',
        'win_length': 5, 'max_placement_dist': 5,
        'min_minutes': 30, 'max_minutes': 90,
        'graduate_pct': 2,
        'num_games': 200, 'num_play_workers': 4, 'max_moves': 100,
        'temperature': 1.5, 'lr': 0.0005,
        'max_dim': 55,
        'seed_random_games': 5000,
    },
    # Consolidation: cycle back to where we get wins to strengthen the model
    {
        'name': 'Phase F: consolidate dist=4',
        'win_length': 5, 'max_placement_dist': 4,
        'min_minutes': 30, 'max_minutes': 45,
        'graduate_pct': 5,
        'num_games': 200, 'num_play_workers': 4, 'max_moves': 90,
        'temperature': 1.5, 'lr': 0.0005,
        'max_dim': 45,
        'seed_random_games': 3000,
    },
    {
        'name': 'Phase G: consolidate dist=5',
        'win_length': 5, 'max_placement_dist': 5,
        'min_minutes': 30, 'max_minutes': 45,
        'graduate_pct': 3,
        'num_games': 200, 'num_play_workers': 4, 'max_moves': 100,
        'temperature': 1.5, 'lr': 0.0005,
        'max_dim': 55,
        'seed_random_games': 3000,
    },
    {
        'name': 'Phase H: push dist=6',
        'win_length': 5, 'max_placement_dist': 6,
        'min_minutes': 30, 'max_minutes': 60,
        'graduate_pct': 1,
        'num_games': 80, 'num_play_workers': 4, 'max_moves': 80,
        'temperature': 1.5, 'lr': 0.0003,
        'max_dim': 55,
        'seed_random_games': 2000,
    },
    {
        'name': 'Phase I: push dist=8 (full rules)',
        'win_length': 5, 'max_placement_dist': 8,
        'min_minutes': 20, 'max_minutes': 45,
        'graduate_pct': 1,
        'num_games': 40, 'num_play_workers': 4, 'max_moves': 80,
        'temperature': 1.5, 'lr': 0.0003,
        'max_dim': 60,
        'seed_random_games': 1000,
    },
]


# ============================================================================
# Batched NN self-play
# ============================================================================
def _nn_worker(model_state_dict, model_config, game_kwargs, num_games,
               max_moves, temperature, max_dim, device_str):
    """Worker process: loads model, plays games, returns winning data."""
    # Set below-normal priority so game processes always get CPU first
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
    except Exception:
        pass
    device = torch.device(device_str)
    game = HeXOGame(**game_kwargs)
    model = ResNet(game, **model_config, device=device)
    model.load_state_dict(model_state_dict)
    model.eval()

    # Use the proven sequential approach
    from overnight_train import nn_fast_selfplay

    # In low-impact mode, play in small chunks with sleeps to limit CPU usage
    low_impact = os.environ.get('LOW_IMPACT', '0') == '1'
    if low_impact:
        all_samples, total_wins, total_played = [], 0, 0
        chunk = 5  # play 5 games at a time
        for i in range(0, num_games, chunk):
            n = min(chunk, num_games - i)
            s, w, p = nn_fast_selfplay(model, game, n, max_moves, temperature, device)
            all_samples.extend(s)
            total_wins += w
            total_played += p
            time.sleep(1.0)  # Throttle: ~0.5s compute per chunk, 1s sleep = ~33% CPU
        samples = all_samples
        wins = total_wins
        played = total_played
    else:
        samples, wins, played = nn_fast_selfplay(
            model, game, num_games, max_moves, temperature, device)

    # Filter oversized
    samples = [s for s in samples if s[0].shape[1] <= max_dim and s[0].shape[2] <= max_dim]
    return samples, wins, played


def parallel_nn_selfplay(model, game_kwargs, model_config, num_games_total,
                         max_moves, temperature, device, max_dim, num_workers=4):
    """Run NN self-play across multiple processes, each with a model copy."""
    model.eval()
    state_dict = model.state_dict()

    # Low-impact: single worker on CPU to minimize resource usage
    if LOW_IMPACT:
        num_workers = 1
        worker_device = 'cpu'
    else:
        worker_device = str(device)

    games_per_worker = num_games_total // num_workers

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = []
        for _ in range(num_workers):
            f = pool.submit(_nn_worker, state_dict, model_config,
                            game_kwargs, games_per_worker, max_moves,
                            temperature, max_dim, worker_device)
            futures.append(f)

        all_samples = []
        total_wins = 0
        total_played = 0
        for f in futures:
            samples, wins, played = f.result()
            all_samples.extend(samples)
            total_wins += wins
            total_played += played

    return all_samples, total_wins, total_played


# ============================================================================
# MCTS refinement phase (reuses existing infrastructure)
# ============================================================================
def run_mcts_phase(phase, model, device, save_dir, phase_start, global_start):
    """Final MCTS refinement using existing parallel infrastructure."""
    from alphazero.self_play import self_play_worker
    from alphazero.inference import InferenceServer

    name = phase['name']
    win_length = phase['win_length']
    dist = phase['max_placement_dist']
    duration_s = phase['duration_minutes'] * 60
    num_workers = phase.get('num_workers', 8)

    game = HeXOGame(max_placement_dist=dist, win_length=win_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=phase['lr'], weight_decay=1e-4)
    args = {
        'C': 2, 'num_searches': phase['num_searches'],
        'dirichlet_epsilon': phase['dirichlet_epsilon'],
        'dirichlet_alpha': phase['dirichlet_alpha'],
        'temperature': phase['temperature'],
        'max_moves_per_game': phase['max_moves'],
        'batch_size': 64,
    }
    trainer = AlphaZero(model, optimizer, game, args)

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

    from overnight_train import inference_server_process, continuous_worker
    server_proc = mp.Process(
        target=inference_server_process,
        args=(model_state, model_config, str(device),
              request_queue, response_queues, stop_event, model_update_queue),
        kwargs={'batch_size': min(num_workers * 8, 64)},
    )
    server_proc.start()

    workers = []
    for wid in range(num_workers):
        p = mp.Process(
            target=continuous_worker,
            args=(wid, game_args, mcts_args, request_queue,
                  response_queues[wid], result_queue, stop_event),
        )
        p.start()
        workers.append(p)
    log(f"  MCTS: {num_workers} workers + inference server started")

    import queue as queue_module
    iteration = 0
    total_wins = 0
    total_games = 0
    replay = []

    while (time.time() - phase_start) < duration_s:
        heartbeat(f"mcts iter={iteration}")
        new_samples = []
        wins = 0
        games_collected = 0
        deadline = time.time() + 30

        while time.time() < deadline and (time.time() - phase_start) < duration_s:
            try:
                game_samples = result_queue.get(timeout=1.0)
                new_samples.extend(game_samples)
                if any(abs(s[2]) > 0.5 for s in game_samples):
                    wins += 1
                games_collected += 1
            except queue_module.Empty:
                pass

        if not new_samples:
            continue

        total_wins += wins
        total_games += games_collected
        replay.extend(new_samples)
        if len(replay) > 30000:
            replay = replay[-30000:]

        train_data = new_samples + random.sample(replay, min(len(new_samples), len(replay)))
        model.train()
        for _ in range(3):
            losses = trainer.train(train_data)
        model.eval()
        try:
            model_update_queue.put(model.state_dict())
        except:
            pass

        elapsed = (time.time() - global_start) / 60
        log(f"  MCTS iter {iteration:3d} [{elapsed:.0f}m] "
            f"{wins}/{games_collected}g p={losses['policy_loss']:.4f} v={losses['value_loss']:.4f}")

        if iteration % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"mcts_iter{iteration}.pt"))
        iteration += 1

    stop_event.set()
    for w in workers:
        w.join(timeout=10)
        if w.is_alive():
            w.terminate()
    server_proc.join(timeout=10)
    if server_proc.is_alive():
        server_proc.terminate()

    wr = 100 * total_wins / max(total_games, 1)
    log(f"  MCTS done: {iteration} iters, {total_games} games, {total_wins} wins ({wr:.0f}%)")


# ============================================================================
# NN self-play phase
# ============================================================================
def run_nn_phase(phase, model, device, save_dir, global_start, deadline_ts=None):
    """Run one NN self-play phase."""
    name = phase['name']
    win_length = phase['win_length']
    dist = phase['max_placement_dist']
    min_s = phase.get('min_minutes', 15) * 60
    max_s = phase.get('max_minutes', 60) * 60
    graduate_pct = phase.get('graduate_pct', 5)
    num_games = phase['num_games']
    max_moves = phase['max_moves']
    temperature = phase['temperature']
    max_dim = phase.get('max_dim', 60)
    phase_start = time.time()

    log(f"\n{'#'*70}")
    log(f"# {name}")
    log(f"# win={win_length}, dist={dist}, games/batch={num_games}, "
        f"max_moves={max_moves}, temp={temperature}")
    elapsed_total = (time.time() - global_start) / 60
    log(f"# Total elapsed: {elapsed_total:.0f}min")
    log(f"{'#'*70}\n")

    game = HeXOGame(max_placement_dist=dist, win_length=win_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=phase['lr'], weight_decay=1e-4)
    batch_sz = 16 if LOW_IMPACT else 64
    dummy_args = {'C': 2, 'num_searches': 1, 'batch_size': batch_sz,
                  'dirichlet_epsilon': 0.25, 'dirichlet_alpha': 0.3,
                  'temperature': 1.0, 'max_moves_per_game': max_moves}
    trainer = AlphaZero(model, optimizer, game, dummy_args)

    replay_buffer = []
    max_replay = 50000

    # Seed with random games to bootstrap
    from overnight_train import generate_random_games
    num_seed = phase.get('seed_random_games', 0)
    if LOW_IMPACT and num_seed > 0:
        num_seed = min(num_seed, 1000)  # Reduce CPU-heavy seeding
    if num_seed > 0:
        log(f"  Seeding: {num_seed} random games...")
        t0 = time.time()
        seed_data, seed_wins, seed_total = generate_random_games(game, num_seed, max_moves)
        seed_data = [s for s in seed_data if s[0].shape[1] <= max_dim and s[0].shape[2] <= max_dim]
        log(f"  {len(seed_data)} samples, {seed_wins}/{seed_total} wins, {time.time()-t0:.1f}s")
        if seed_data:
            replay_buffer.extend(seed_data)
            # Pre-train on seed data — cap to keep it fast for large boards
            max_pretrain = min(2000, len(seed_data))
            pretrain = random.sample(seed_data, max_pretrain) if len(seed_data) > max_pretrain else seed_data
            model.train()
            for epoch in range(3):
                losses = trainer.train(pretrain)
            model.eval()
            torch.cuda.empty_cache()
            log(f"  Pre-trained ({max_pretrain} samp): p={losses['policy_loss']:.4f} v={losses['value_loss']:.4f}")
    iteration = 0
    total_wins_phase = 0
    total_games_phase = 0

    games_per_iter = phase.get('num_games', 128)
    if LOW_IMPACT:
        games_per_iter = min(games_per_iter, 100)  # Fewer games, less CPU load
    num_play_workers = phase.get('num_play_workers', 4)
    model_config = {'num_resBlocks': len(model.backBone), 'num_hidden': model.num_hidden}
    game_kwargs = {'max_placement_dist': dist, 'win_length': win_length}

    recent_win_rates = []  # Track last N iterations for graduation

    while True:
        elapsed_s = time.time() - phase_start
        # Check graduation: past min time AND recent win rate above threshold
        if elapsed_s >= min_s and len(recent_win_rates) >= 5:
            avg_wr = sum(recent_win_rates[-5:]) / 5
            if avg_wr >= graduate_pct:
                log(f"  ** Graduated! avg win rate {avg_wr:.1f}% >= {graduate_pct}% over last 5 iters **")
                break
        # Hard deadline
        if deadline_ts and time.time() >= deadline_ts:
            log(f"  ** DEADLINE — saving and stopping **")
            break
        # Hard time cap
        if elapsed_s >= max_s:
            avg_wr = sum(recent_win_rates[-5:]) / 5 if recent_win_rates else 0
            log(f"  ** Time cap reached ({phase.get('max_minutes')}min), avg win rate {avg_wr:.1f}% **")
            break

        heartbeat(f"nn phase={name} iter={iteration}")
        t0 = time.time()

        # Play games in parallel processes
        samples, wins, played = parallel_nn_selfplay(
            model, game_kwargs, model_config, games_per_iter,
            max_moves, temperature, device, max_dim, num_play_workers)

        total_wins_phase += wins
        total_games_phase += played

        if not samples:
            # No wins — increase temperature temporarily
            temperature = min(temperature + 0.1, 3.0)
            log(f"  Iter {iteration:3d}: 0 wins/{played}g, raising temp to {temperature:.1f}")
            recent_win_rates.append(0.0)
            iteration += 1
            continue

        # Lower temperature slightly if we're getting wins
        if wins > played * 0.1:
            temperature = max(temperature - 0.05, 0.8)

        # Replay buffer
        replay_buffer.extend(samples)
        if len(replay_buffer) > max_replay:
            replay_buffer = replay_buffer[-max_replay:]

        # Build training data: new + replay
        train_data = list(samples)
        if len(replay_buffer) > len(samples):
            extra = min(len(samples) * 2, len(replay_buffer))
            train_data.extend(random.sample(replay_buffer, extra))

        # Train
        model.train()
        num_epochs = 2 if LOW_IMPACT else 4
        for ep in range(num_epochs):
            losses = trainer.train(train_data)
            if LOW_IMPACT:
                time.sleep(0.5)  # Let GPU handle game frames
        model.eval()
        if LOW_IMPACT:
            torch.cuda.empty_cache()
            time.sleep(3.0)  # Cooldown between iterations

        iter_time = time.time() - t0
        phase_elapsed = (time.time() - phase_start) / 60
        total_elapsed = (time.time() - global_start) / 60
        wr = 100 * wins / played

        log(f"  Iter {iteration:3d} [{phase_elapsed:5.1f}m / {total_elapsed:5.1f}m] "
            f"{wins}/{played}g ({wr:.0f}%) {len(samples)} samp "
            f"p={losses['policy_loss']:.4f} v={losses['value_loss']:.4f} "
            f"t={temperature:.2f} ({iter_time:.0f}s) [buf={len(replay_buffer)}]")

        recent_win_rates.append(wr)

        if iteration % 5 == 0:
            tag = name.split(':')[0].strip().replace(' ', '').lower()
            torch.save(model.state_dict(), os.path.join(save_dir, f"{tag}_iter{iteration}.pt"))
        iteration += 1

    # Save final
    tag = name.split(':')[0].strip().replace(' ', '').lower()
    torch.save(model.state_dict(), os.path.join(save_dir, f"{tag}_final.pt"))

    phase_time = (time.time() - phase_start) / 60
    wr = 100 * total_wins_phase / max(total_games_phase, 1)
    log(f"\n  Phase done: {iteration} iters, {total_games_phase} games, "
        f"{total_wins_phase} wins ({wr:.0f}%), {phase_time:.1f}min\n")

    return model


# ============================================================================
# Main
# ============================================================================
def main():
    global LOG_PATH, HEARTBEAT_PATH, LOW_IMPACT
    base_dir = os.path.dirname(os.path.abspath(__file__))
    LOG_PATH = os.path.join(base_dir, 'nn_train.log')
    HEARTBEAT_PATH = os.path.join(base_dir, 'heartbeat.txt')
    LOW_IMPACT = os.environ.get('LOW_IMPACT', '0') == '1'
    if LOW_IMPACT:
        try:
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
        except Exception:
            pass

    with open(LOG_PATH, 'w') as f:
        f.write("")

    mp.set_start_method('spawn', force=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    if LOW_IMPACT:
        log(f"*** LOW IMPACT MODE: 1 CPU worker, batch=32, 2 epochs, low priority ***")
    log(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Approach: NN self-play (batched GPU) -> MCTS refinement")

    save_dir = 'checkpoints_nn'
    os.makedirs(save_dir, exist_ok=True)

    # Load pretrained checkpoint
    model = ResNet(HeXOGame(), num_resBlocks=10, num_hidden=128, device=device)
    checkpoint = os.environ.get('CHECKPOINT', 'checkpoints_overnight/phase3_final.pt')
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        log(f"Loaded checkpoint: {checkpoint}")
    else:
        log(f"WARNING: checkpoint not found: {checkpoint}, starting fresh")

    total_planned = sum(p.get('max_minutes', p.get('duration_minutes', 30)) for p in PHASES)
    log(f"Phases: {len(PHASES)}, planned: {total_planned}min ({total_planned/60:.1f}h)")
    log(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Skip phases via env var
    start_phase = int(os.environ.get('START_PHASE', '1'))
    phases = PHASES[start_phase - 1:]

    # Hard deadline: stop by this time no matter what
    deadline_str = os.environ.get('DEADLINE', '')
    if deadline_str:
        # Parse HH:MM format, assume today or tomorrow
        h, m = map(int, deadline_str.split(':'))
        import datetime
        now = datetime.datetime.now()
        deadline_dt = now.replace(hour=h, minute=m, second=0)
        if deadline_dt <= now:
            deadline_dt += datetime.timedelta(days=1)
        DEADLINE_TS = deadline_dt.timestamp()
        log(f"Hard deadline: {deadline_dt.strftime('%Y-%m-%d %H:%M')}")
    else:
        DEADLINE_TS = None

    global_start = time.time()
    heartbeat("starting")

    for phase in phases:
        if DEADLINE_TS and time.time() >= DEADLINE_TS:
            log(f"\n  *** DEADLINE REACHED — stopping ***")
            break
        if phase.get('mcts_phase'):
            phase_start = time.time()
            log(f"\n{'#'*70}")
            log(f"# {phase['name']} (MCTS REFINEMENT)")
            log(f"{'#'*70}\n")
            run_mcts_phase(phase, model, device, save_dir, phase_start, global_start)
        else:
            model = run_nn_phase(phase, model, device, save_dir, global_start, DEADLINE_TS)

    torch.save(model.state_dict(), os.path.join(save_dir, 'model_final.pt'))
    total_time = (time.time() - global_start) / 60
    log(f"\n{'='*60}")
    log(f"DONE: {total_time:.0f}min ({total_time/60:.1f}h)")
    log(f"Final model: {save_dir}/model_final.pt")
    log(f"{'='*60}")


if __name__ == '__main__':
    main()
