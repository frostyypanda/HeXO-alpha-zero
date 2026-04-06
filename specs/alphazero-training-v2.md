# Feature: AlphaZero Training v2 (Gumbel + Warm-Start)

## Status
draft

## Why this exists

AlphaZero self-play is the proven path to superhuman game AI, but it requires:
1. A competent initial model (solved by heuristic bootstrap — see imitation-learning.md)
2. Efficient use of compute (solved by Gumbel MCTS + KataGo techniques)

Prior runs wasted 14+ hours of GPU time because: (a) cold-start with random NN produced
0 wins, and (b) standard PUCT MCTS at 200-800 sims was too expensive per game, producing
too few training games per hour.

This spec describes the self-play pipeline that starts from the imitation-learned model
and uses Gumbel MCTS + playout cap randomization + progressive simulation to train
efficiently.

## What it does

### Gumbel MCTS (replaces standard PUCT)

Standard PUCT MCTS fails at low simulation counts because it can't reliably improve the
policy with only 20-50 simulations across 300+ legal moves. Gumbel MCTS (DeepMind 2022)
fixes this by:

1. **Sample top-k moves from Gumbel-perturbed log-priors** — instead of exploring all
   moves via UCB, only the top-k (e.g., 16-32) most promising moves are considered
2. **Sequential Halving** — among the top-k, progressively eliminate the weaker half by
   allocating simulations. After log2(k) rounds, only the best move remains
3. **Guaranteed policy improvement** — the completed Q-values are provably better than
   the raw policy, even with very few total simulations

**Why this matters for HeXO:** With 300+ legal moves, standard MCTS at 50 sims spreads
simulations too thin. Gumbel focuses all simulations on the ~16-32 moves the policy
thinks are promising, then picks the best among those. This produces reliable policy
improvement at 16-50 total simulations — 4-10x fewer than standard PUCT needs.

### Playout Cap Randomization (KataGo)

Not every training position needs deep search:

- **25% of moves**: Full search (all simulations). Produces high-quality policy targets.
- **75% of moves**: Quick search (1/4 of simulations). Produces more games per hour.

Both contribute to value training (game outcomes are the same). Only the full-search
positions contribute policy targets. This decouples:
- **Policy learning** — needs deep search but few positions
- **Value learning** — needs many games but shallow search is fine

Net effect: ~1.4x more training data per GPU-hour for free.

### Progressive Simulation Schedule

Start training with low simulation counts and increase as the model strengthens:

| Training phase | Simulations | Gumbel top-k | Rationale |
|---------------|-------------|--------------|-----------|
| Iterations 1-50 | 32 | 16 | Model is still crude. Fast games = more data = faster learning |
| Iterations 51-100 | 64 | 16 | Model is competent. Moderate search for better policy targets |
| Iterations 101+ | 128-200 | 32 | Model is strong. Deep search for refined policy |

The schedule auto-adapts: if arena win rate vs previous best exceeds 60%, increase sims.
If arena rejects 5 consecutive models, decrease sims (more data, less quality per game).

### Training Loop

```
for each iteration:
    1. Parallel self-play (Gumbel MCTS + current NN) → generate games
       - Apply playout cap randomization (25% full / 75% quick)
    2. Add ALL games to replay buffer (wins, losses, AND draws)
    3. Train NN on replay buffer samples
       - Policy target: Gumbel-completed visit distribution (full-search positions only)
       - Value target: game outcome (+1/-1/0)
    4. Arena: pit new model vs previous best (using full simulations)
    5. Keep stronger model, checkpoint
    6. Adjust simulation count if needed (progressive schedule)
```

### Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| win_length | 5 | Training target (scale to 6 later) |
| max_placement_dist | 8 | Full rules from the start |
| gumbel_top_k | 16 (early) → 32 (late) | Focus simulations on promising moves |
| num_searches | 32 → 200 (progressive) | Start fast, increase quality over time |
| playout_cap_fraction | 0.25 | 25% full search, 75% quick search |
| num_workers | 8 | Good utilization of 7800X3D |
| batch_size | 128 | Stable gradients |
| lr | 0.001 → 0.0001 | Cosine decay |
| replay_buffer_size | 100000 | ~500 games of data |
| arena_games | 40 | Sufficient to detect improvement |
| arena_threshold | 0.55 | Accept if >55% win rate |
| temperature | 1.0 (moves 1-30), 0.3 (after) | Explore early, exploit late |
| dirichlet_epsilon | 0.25 | Standard root noise |
| dirichlet_alpha | 0.15 | Lower for large action space |

### Train on ALL Games

**Critical change from prior approach.** All games are kept:
- Won games: positions labeled +1 (winner) / -1 (loser)
- Drawn games: positions labeled 0

This provides 10-100x more data than win-only filtering. Draws teach the value head that
most positions are close to 0, with wins/losses being the exception.

### Monitoring & Safeguards

Each iteration logs:
- Win rate (% decisive games), average game length
- Policy loss, value loss
- Arena results (new vs best)
- Current simulation count, games per hour

**Automatic safeguards:**
- If 0 decisive games for 5 iterations → roll back to last good checkpoint + lower sims
- If arena rejects 10 consecutive models → plateau detected, try LR bump or sim increase
- Max board dimension filter (60px) prevents VRAM spikes

### Scaling to win=6

After model plays well at win=5 (beats strong heuristic >90%):
1. Load win=5 best model
2. Switch to win=6 rules
3. Continue self-play — spatial patterns transfer well for +1 win length

## Boundaries & edge cases

**What it intentionally does NOT do:**
- No curriculum — single difficulty throughout (full rules)
- No auxiliary targets in v2.0 (add later if model plateaus)
- No hex symmetry augmentation in v2.0 (add later if data diversity is bottleneck)
- No distributed training — single machine

**Known limitations:**
- Progressive simulation needs monitoring (auto-schedule may need manual override)
- Gumbel top-k=16 may miss good moves in open positions (mitigated by Dirichlet noise)
- Large boards (dist=8) can cause VRAM spikes — mitigated by max_dim filter

## Testing & verification

### Key scenarios
- [ ] Gumbel MCTS at 32 sims produces better policy than PUCT at 32 sims (on 100 positions)
- [ ] First self-play iteration produces >30% decisive games
- [ ] Policy loss decreases over first 20 iterations
- [ ] Games per hour: >10 at 32 sims with 8 workers (throughput check)
- [ ] New model beats imitation model in arena within 50 iterations
- [ ] Model at iteration 100+ beats strong heuristic >80%

### Edge cases
- Self-play produces all draws: increase temperature, decrease sims (more exploration)
- VRAM OOM on large boards: reduce batch size or max_dim
- Gumbel top-k too small (misses good moves): increase k, check with Dirichlet noise
- Workers die mid-game: existing health check handles this

### Automation notes
- Automated: iteration loop, checkpointing, arena, progressive sim schedule
- Manual checkpoints: visual inspection at iterations 0, 50, 100
- End condition: N iterations or 10 consecutive arena rejections (plateau)

## Decision log
| Date | Decision | Why | Who |
|------|----------|-----|-----|
| 2026-04-06 | Gumbel MCTS over standard PUCT | Standard PUCT fails at <100 sims for 300+ action spaces. Gumbel guarantees improvement at 16-50 sims. 3-5x speedup | Kei + Claude |
| 2026-04-06 | Playout cap randomization | KataGo showed 1.37x speedup. Decouples policy learning (needs deep search) from value learning (needs many games). Free improvement | Kei + Claude |
| 2026-04-06 | Progressive simulation schedule | Early model doesn't benefit from deep search. Start fast (more games), increase quality as model improves. Validated by MiniZero 2024 | Kei + Claude |
| 2026-04-06 | Start at 32 sims (not 200) | With Gumbel, 32 sims is productive. At 8 workers, this generates ~15+ games/hour vs ~3/hour at 200 sims. 5x more training data per hour | Kei + Claude |
| 2026-04-06 | Train on all games (not just wins) | Prior approach: 0.3% win rate → ~50 training samples per iteration. All games: ~5000 samples. 100x more signal | Kei + Claude |
| 2026-04-06 | No auxiliary targets initially | KataGo's ownership targets give 1.65x but add implementation complexity. Get baseline working first, add if model plateaus | Kei + Claude |

## Related
- [specs/_overview.md](_overview.md) — epic context
- [specs/001-approach-selection.md](001-approach-selection.md) — why this approach
- [specs/imitation-learning.md](imitation-learning.md) — produces the starting model
- [specs/heuristic-player.md](heuristic-player.md) — bootstrap data source
- `alphazero/mcts.py` — will be rewritten for Gumbel selection
- `alphazero/pipeline.py` — modified for progressive sims, playout cap
