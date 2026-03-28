"""
Entry point for HeXO AlphaZero training.

Usage:
    python train_hexo.py                          # parallel training (default)
    python train_hexo.py --single-threaded        # single-threaded
    python train_hexo.py --num-workers 4          # 4 parallel workers
    python train_hexo.py --resume 5               # resume from iteration 5
    python train_hexo.py --num-searches 200       # fewer MCTS sims (faster)
"""

import argparse
from config import get_default_args
from alphazero.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train HeXO AlphaZero")
    parser.add_argument('--single-threaded', action='store_true',
                        help='Disable parallel self-play')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel self-play workers')
    parser.add_argument('--num-iterations', type=int, default=None)
    parser.add_argument('--num-self-play', type=int, default=None,
                        help='Games per iteration')
    parser.add_argument('--num-searches', type=int, default=None,
                        help='MCTS simulations per move')
    parser.add_argument('--num-epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--num-res-blocks', type=int, default=None)
    parser.add_argument('--num-hidden', type=int, default=None)
    parser.add_argument('--resume', type=int, default=None,
                        help='Resume from iteration N')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--save-dir', type=str, default=None)
    args = parser.parse_args()

    config = get_default_args()

    # Apply CLI overrides
    if args.single_threaded:
        config['parallel'] = False
    if args.num_workers is not None:
        config['num_workers'] = args.num_workers
    if args.num_iterations is not None:
        config['num_iterations'] = args.num_iterations
    if args.num_self_play is not None:
        config['num_selfPlay_iterations'] = args.num_self_play
    if args.num_searches is not None:
        config['num_searches'] = args.num_searches
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['lr'] = args.lr
    if args.num_res_blocks is not None:
        config['num_resBlocks'] = args.num_res_blocks
    if args.num_hidden is not None:
        config['num_hidden'] = args.num_hidden
    if args.resume is not None:
        config['resume_iteration'] = args.resume
    if args.cpu:
        config['device'] = 'cpu'
    if args.save_dir is not None:
        config['save_dir'] = args.save_dir

    run_pipeline(config)


if __name__ == '__main__':
    main()
