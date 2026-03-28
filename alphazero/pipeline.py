"""
Parallel AlphaZero training pipeline for HeXO.

Orchestrates:
  1. GPU inference server process — batches NN evaluations.
  2. CPU self-play workers — run MCTS games in parallel.
  3. Training process — trains on collected data.
  4. Arena evaluation — pits new model vs previous best.

Designed for: AMD 7800X3D (16 threads), 32GB RAM, RTX 4090 (24GB VRAM).
"""

import os
import copy
import torch
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm

from hexo.game import HeXOGame
from alphazero.model import ResNet
from alphazero.mcts import MCTS
from alphazero.train import AlphaZero
from alphazero.inference import InferenceServer
from alphazero.self_play import self_play_worker
from alphazero.arena import evaluate_models


def run_inference_server(model_state_dict, model_config, device_str,
                         request_queue, response_queues, stop_event,
                         batch_size=32):
    """Inference server process entry point."""
    device = torch.device(device_str)
    game = HeXOGame()
    model = ResNet(game, **model_config, device=device)
    model.load_state_dict(model_state_dict)
    model.eval()

    server = InferenceServer(model, device, batch_size=batch_size)
    server.run(request_queue, response_queues, stop_event)


def parallel_self_play(game, model, args, num_games, num_workers=8):
    """
    Run parallel self-play using multiprocessing.

    Launches an inference server + multiple self-play workers.
    Returns list of training samples.
    """
    mp.set_start_method('spawn', force=True)

    device_str = str(model.device)
    model_state = model.state_dict()
    model_config = {
        'num_resBlocks': len(model.backBone),
        'num_hidden': model.num_hidden,
    }

    request_queue = mp.Queue()
    response_queues = {}
    result_queue = mp.Queue()
    stop_event = mp.Event()

    # Create per-worker response queues
    for wid in range(num_workers):
        response_queues[wid] = mp.Queue()

    # Distribute games across workers
    games_per_worker = num_games // num_workers
    remainder = num_games % num_workers

    game_args = {'max_placement_dist': game.max_placement_dist}
    mcts_args = {
        'C': args.get('C', 2),
        'num_searches': args.get('num_searches', 60),
        'dirichlet_epsilon': args.get('dirichlet_epsilon', 0.25),
        'dirichlet_alpha': args.get('dirichlet_alpha', 0.3),
        'temperature': args.get('temperature', 1.25),
    }

    # Start inference server
    server_proc = mp.Process(
        target=run_inference_server,
        args=(model_state, model_config, device_str,
              request_queue, response_queues, stop_event),
        kwargs={'batch_size': min(num_workers * 2, 64)},
    )
    server_proc.start()

    # Start workers
    workers = []
    for wid in range(num_workers):
        n = games_per_worker + (1 if wid < remainder else 0)
        p = mp.Process(
            target=self_play_worker,
            args=(wid, game_args, mcts_args, request_queue,
                  response_queues[wid], result_queue, n, stop_event),
        )
        p.start()
        workers.append(p)

    # Collect results
    memory = []
    total_expected = num_games
    collected = 0

    with tqdm(total=total_expected, desc="Parallel self-play") as pbar:
        while collected < total_expected:
            try:
                samples = result_queue.get(timeout=300)
                memory.extend(samples)
                collected += 1
                pbar.update(1)
            except Exception:
                # Check if workers are still alive
                alive = any(w.is_alive() for w in workers)
                if not alive:
                    break

    # Shutdown
    stop_event.set()
    for w in workers:
        w.join(timeout=10)
    server_proc.join(timeout=10)

    return memory


def run_pipeline(args):
    """
    Full parallel AlphaZero training pipeline.

    Args:
        args: dict with all configuration. See config.py for defaults.
    """
    device = torch.device(args.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    game = HeXOGame(max_placement_dist=args.get('max_placement_dist', 8))

    # Initialize model
    model = ResNet(
        game,
        num_resBlocks=args.get('num_resBlocks', 10),
        num_hidden=args.get('num_hidden', 128),
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=args.get('lr', 0.001),
                                  weight_decay=args.get('weight_decay', 1e-4))

    save_dir = args.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    # Load checkpoint if resuming
    resume_iter = args.get('resume_iteration', None)
    if resume_iter is not None:
        model_path = os.path.join(save_dir, f"model_{resume_iter}.pt")
        optim_path = os.path.join(save_dir, f"optimizer_{resume_iter}.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))
        optimizer.load_state_dict(torch.load(optim_path, map_location=device))
        print(f"Resumed from iteration {resume_iter}")
        start_iter = resume_iter + 1
    else:
        start_iter = 0

    # Training params
    num_iterations = args.get('num_iterations', 20)
    num_self_play = args.get('num_selfPlay_iterations', 100)
    num_epochs = args.get('num_epochs', 4)
    num_workers = args.get('num_workers', 8)
    use_parallel = args.get('parallel', True)
    arena_games = args.get('arena_games', 20)
    arena_threshold = args.get('arena_threshold', 0.55)

    best_model_state = copy.deepcopy(model.state_dict())
    trainer = AlphaZero(model, optimizer, game, args)

    for iteration in range(start_iter, num_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")

        # --- Self-play ---
        model.eval()
        if use_parallel and num_workers > 1:
            print(f"Parallel self-play: {num_self_play} games, {num_workers} workers")
            memory = parallel_self_play(game, model, args, num_self_play, num_workers)
        else:
            print(f"Single-threaded self-play: {num_self_play} games")
            memory = []
            for _ in tqdm(range(num_self_play), desc="Self-play"):
                memory += trainer.self_play()

        print(f"Collected {len(memory)} training samples.")

        # --- Training ---
        model.train()
        for epoch in range(num_epochs):
            losses = trainer.train(memory)
            print(f"  Epoch {epoch + 1}/{num_epochs} — "
                  f"policy: {losses['policy_loss']:.4f}, "
                  f"value: {losses['value_loss']:.4f}")

        # --- Arena evaluation ---
        if arena_games > 0 and iteration > 0:
            old_model = ResNet(
                game,
                num_resBlocks=args.get('num_resBlocks', 10),
                num_hidden=args.get('num_hidden', 128),
                device=device,
            )
            old_model.load_state_dict(best_model_state)

            is_better = evaluate_models(
                game, model, old_model, args,
                num_games=arena_games,
                win_threshold=arena_threshold,
            )

            if is_better:
                print("New model accepted!")
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                print("New model rejected — reverting to previous best.")
                model.load_state_dict(best_model_state)
        else:
            best_model_state = copy.deepcopy(model.state_dict())

        # --- Checkpoint ---
        model_path = os.path.join(save_dir, f"model_{iteration}.pt")
        optim_path = os.path.join(save_dir, f"optimizer_{iteration}.pt")
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optim_path)
        print(f"Saved: {model_path}")

    # Save final best
    final_path = os.path.join(save_dir, "model_best.pt")
    torch.save(best_model_state, final_path)
    print(f"\nTraining complete. Best model: {final_path}")
