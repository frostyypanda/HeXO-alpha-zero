"""Smoke test: run overnight_train with very short durations."""
import os
import sys
import time

# Monkey-patch the phases to be very short
import overnight_train
overnight_train.PHASES = [
    {
        'name': 'Smoke 1: win=3, dist=2',
        'win_length': 3,
        'max_placement_dist': 2,
        'duration_minutes': 2,
        'num_searches': 25,
        'max_moves_per_game': 30,
        'games_per_iter': 5,
        'seed_random_games': 200,
        'seed_epochs': 2,
        'temperature': 1.5,
        'dirichlet_epsilon': 0.5,
        'dirichlet_alpha': 0.5,
        'lr': 0.001,
    },
    {
        'name': 'Smoke 2: win=4, dist=3',
        'win_length': 4,
        'max_placement_dist': 3,
        'duration_minutes': 2,
        'num_searches': 40,
        'max_moves_per_game': 50,
        'games_per_iter': 3,
        'seed_random_games': 500,
        'seed_epochs': 2,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.35,
        'dirichlet_alpha': 0.3,
        'lr': 0.001,
    },
]

overnight_train.main()
