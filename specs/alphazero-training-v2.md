# Feature: AlphaZero Training v2 (Warm-Start)

## Status
draft

## Why this exists

AlphaZero self-play is the proven path to superhuman game AI, but it requires a
competent initial model to guide MCTS. Prior runs showed that starting from a random
model leads to cold-start collapse (0 wins → no signal → model stays random).

This spec describes the self-play training pipeline that starts from the
imitation-learned model and refines it through self-play. This is the phase where the
model surpasses its heuristic teacher and develops novel strategies.

## What it does

### Training Loop

Standard AlphaZero loop, using the existing parallel infrastructure:

```
for each iteration:
    1. Parallel self-play (MCTS + current NN) → generate games
    2. Add games to replay buffer
    3. Train NN on replay buffer samples
    4. Arena: pit new model vs previous best
    5. Keep stronger model, checkpoint
```

Uses `alphazero/pipeline.py`'s `run_pipeline()` with these modifications:
- **Start from imitation checkpoint** (not random weights)
- **Full rules from the start** (win=5, dist=8, no curriculum)
- **Train on ALL games** (not just wins)
- **Larger replay buffer** (100k samples)

### Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| win_length | 5 | Training target (6 for official HeXO, but 5 is faster to validate) |
| max_placement_dist | 8 | Full rules |
| num_searches | 200-400 | Balance quality vs speed. Start at 200, increase if model stalls |
| num_workers | 8 | Good utilization of 7800X3D (16 threads, leave headroom) |
| batch_size | 128 | Large batches for stable training |
| lr | 0.001 → 0.0001 | Cosine decay over training run |
| replay_buffer_size | 100000 | ~500 games worth of data |
| arena_games | 40 | Enough to detect improvement with confidence |
| arena_threshold | 0.55 | Accept model if >55% win rate vs previous |
| temperature | 1.0 (moves 1-30), 0.3 (moves 31+) | Explore early, exploit late |
| dirichlet_epsilon | 0.25 | Standard AlphaZero root noise |
| dirichlet_alpha | 0.15 | Lower alpha for larger action space (Go uses 0.03 for 19x19) |

### Training on All Games (Not Just Wins)

**Critical change from prior approach:** All games are kept, including draws.

For won games: positions labeled +1 (winner) / -1 (loser)
For drawn games: all positions labeled 0

This provides 10-100x more training data than the win-only approach and teaches the
value head the full range of position evaluations.

### Monitoring & Diagnostics

Each iteration logs:
- Win rate (% of self-play games that end decisively)
- Average game length
- Policy loss, value loss
- Arena results (new vs best)
- MCTS statistics (avg simulations, avg tree depth)

**Warning signs that need intervention:**
- Win rate drops to 0% for 10+ iterations → model collapsed, roll back to last good checkpoint
- Policy loss increases steadily → learning rate too high or data quality issue
- Arena consistently rejects new model → model plateaued, try increasing simulations or lr

### Checkpointing Strategy

- Save every 5 iterations: `model_iter{N}.pt`
- Save after each arena acceptance: `model_best.pt`
- Keep last 3 best models for rollback safety
- Save optimizer state for resuming interrupted runs

### Scaling Up to win=6 (Official HeXO)

After the model plays well at win=5:
1. Load the win=5 model
2. Switch to win=6 rules
3. Continue self-play training

This is the one curriculum step that SHOULD work because:
- The spatial patterns learned at win=5 are directly relevant to win=6
- The model already understands line-building, blocking, forking
- win=6 is only 1 step harder than win=5 (not 2-3 steps like prior curriculum)
- The model starts competent enough that MCTS can find wins

## Boundaries & edge cases

**What it intentionally does NOT do:**
- No curriculum within this phase — single difficulty throughout
- No NN-only fallback — always uses MCTS for self-play
- No model architecture changes — same ResNet as before
- No distributed training — single machine (RTX 4090 + 7800X3D)

**Known limitations:**
- Training time scales with game complexity (longer games = fewer iterations per hour)
- At full rules (dist=8), boards can grow large, causing VRAM spikes on batching
  - Mitigated by max_dim filter (drop states >60x60) and batch size tuning
- Parallel self-play throughput depends on MCTS simulation count
  - 200 sims: ~2-5 games/min across 8 workers
  - 400 sims: ~1-3 games/min

**Hard requirement:**
- The imitation-learned model MUST pass validation gates before starting self-play
  (see imitation-learning.md). Starting self-play with a bad model wastes hours.

## Testing & verification

### Key scenarios
- [ ] First self-play iteration produces >30% decisive games (not all draws)
- [ ] Policy loss decreases over first 20 iterations
- [ ] New model beats imitation model in arena within 50 iterations
- [ ] Model at iteration 100+ beats strong heuristic player >80%
- [ ] Self-play games show recognizable strategic patterns (line extension, blocking, forks)

### Edge cases
- Self-play produces all draws: model still too weak. Increase MCTS sims or re-do imitation
- Arena always rejects new model: model plateaued. Increase exploration (dirichlet, temp)
- VRAM OOM on large boards: reduce batch size or max_dim filter
- Workers die mid-game: existing health check + restart logic in pipeline.py handles this

### Automation notes
- Automated: iteration loop, checkpointing, arena evaluation
- Manual checkpoints: visual inspection of self-play games at iterations 0, 50, 100
- End condition: run for N iterations or until arena plateau (10 consecutive rejections)

## Decision log
| Date | Decision | Why | Who |
|------|----------|-----|-----|
| 2026-04-06 | Start at 200 sims (not 800) | Prior runs used 800 but were bottlenecked on game generation speed. 200 balances quality and throughput. Can increase later | Kei + Claude |
| 2026-04-06 | Train on all games (not just wins) | The #1 failure mode was insufficient training data from win-only filtering. Draws teach the value head too | Kei + Claude |
| 2026-04-06 | No curriculum, full rules from start | Curriculum failed due to transfer gaps. Imitation bootstrap makes curriculum unnecessary | Kei + Claude |
| 2026-04-06 | Lower dirichlet_alpha (0.15 vs 0.3) | HeXO has ~300+ legal moves vs 361 for Go. Lower alpha = flatter noise = more exploration across large action space | Kei + Claude |
| 2026-04-06 | win=5 first, then scale to win=6 | win=5 is faster to iterate on. win=5→6 is the one curriculum step with strong transfer (same spatial patterns, just 1 longer) | Kei + Claude |

## Related
- [specs/_overview.md](_overview.md) — epic context
- [specs/imitation-learning.md](imitation-learning.md) — produces the starting model
- `alphazero/pipeline.py` — `run_pipeline()` orchestrator
- `alphazero/self_play.py` — parallel MCTS workers
- `alphazero/inference.py` — batched GPU inference server
- `alphazero/arena.py` — model evaluation
- `config.py` — default hyperparameters (will be updated)
