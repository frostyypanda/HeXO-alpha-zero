"""
Default hyperparameters and configuration for HeXO AlphaZero.

Usage:
    from config import get_default_args
    args = get_default_args()
    args['num_iterations'] = 50  # override as needed
"""


def get_default_args():
    """Return default configuration dict."""
    return {
        # --- Game ---
        'max_placement_dist': 8,
        'win_length': 5,               # in-a-row to win (5 for training, 6 for official HeXO)

        # --- Model ---
        'num_resBlocks': 10,
        'num_hidden': 128,

        # --- MCTS ---
        'C': 2,                        # UCB exploration constant
        'num_searches': 800,           # simulations per move (training)
        'dirichlet_epsilon': 0.25,     # root noise weight (0 = off)
        'dirichlet_alpha': 0.3,        # Dirichlet concentration
        'temperature': 1.25,           # action sampling temperature

        # --- Training ---
        'num_iterations': 20,          # outer training iterations
        'num_selfPlay_iterations': 100,  # games per iteration
        'num_epochs': 4,               # training epochs per iteration
        'batch_size': 64,              # training batch size
        'lr': 0.001,                   # learning rate
        'weight_decay': 1e-4,          # L2 regularization

        # --- Parallelization ---
        'parallel': True,
        'num_workers': 8,              # CPU self-play workers

        # --- Arena ---
        'arena_games': 20,             # games for model evaluation
        'arena_threshold': 0.55,       # win rate to accept new model
        'arena_num_searches': 60,      # MCTS sims during arena

        # --- Paths ---
        'save_dir': 'checkpoints',
        'device': 'cuda',              # 'cuda' or 'cpu'

        # --- Resume ---
        'resume_iteration': None,      # set to int to resume
    }
