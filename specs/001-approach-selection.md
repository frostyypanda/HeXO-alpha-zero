# Decision 001: Training Approach Selection

## Status
accepted

## Context

Multiple AlphaZero training runs (Mar 28-30, 2026) failed due to cold-start collapse:
random NN misleads MCTS, producing 0 wins, no gradient signal. The overnight curriculum
approach and NN-only curriculum both failed at win>=4.

We evaluated all viable approaches for training a HeXO AI on a single RTX 4090.

## Options considered

### 1. AlphaZero + Heuristic Bootstrap (selected)
Pre-train NN on heuristic self-play data, then standard AlphaZero MCTS self-play.

**Pros:** Eliminates cold-start. Proven approach (AlphaGo v1 did this with human games).
Well-supported by literature. Reuses our entire existing infrastructure.
**Cons:** Bounded by heuristic quality initially. Requires building the heuristic.

### 2. PPO / Policy Gradient (rejected)
Direct RL without tree search.

**Pros:** Simpler training loop, faster per iteration.
**Cons:** Significantly weaker than MCTS-guided approaches for board games. No top
Gomoku/Go agent uses PPO alone. Can't handle 300+ action space efficiently. Sparse
rewards over 20-40 move games make credit assignment extremely hard.

### 3. Gumbel AlphaZero (incorporated as enhancement)
Replaces PUCT selection with Gumbel-based sampling that guarantees policy improvement
even with very few simulations (as low as 2).

**Pros:** 3-5x speedup by enabling productive training at 50 sims vs 200+. Proven on
Go, Othello, Chess (DeepMind 2022, MiniZero 2024).
**Cons:** Requires MCTS selection rewrite.
**Decision:** Incorporate as an enhancement to option 1, not a standalone approach.

### 4. Expert Iteration / ExIt (equivalent to AlphaZero)
ExIt and AlphaZero share the same fundamental framework (UCL thesis confirmed this).
No meaningful advantage or disadvantage vs AlphaZero for our use case.

### 5. Pure MCTS without NN (rejected as final agent)
**Pros:** No training needed.
**Cons:** Far weaker than NN-guided MCTS. In GomoCup competitions, pure MCTS agents
rank well below neural-network-enhanced agents.

### 6. NNUE + Alpha-Beta (Rapfi's approach) (deferred)
GomoCup 2024 champion uses NNUE + alpha-beta. Beat KataGomo under limited compute.

**Pros:** Extremely efficient inference. Strong for Gomoku-like games.
**Cons:** Requires training a CNN teacher first (same upfront cost), then distilling to
NNUE. Pattern codebook design is game-specific and complex.
**Decision:** Consider as a later optimization for fast interactive play, not for initial
training.

### 7. KataGo Techniques (incorporated as enhancements)
KataGo achieved ~50x compute reduction vs ELF OpenGo through specific techniques.

Key applicable techniques:
- Playout cap randomization: 1.37x speedup (trivial to implement)
- Auxiliary ownership targets: 1.65x speedup (moderate effort)
- Game-specific input features: 1.55x (moderate effort)
- Global pooling in residual blocks: 1.60x (already partially used in value head)

**Decision:** Incorporate highest-impact techniques into training pipeline.

## Decision

**AlphaZero with heuristic bootstrap** as the core approach, enhanced with:
1. **Gumbel MCTS selection** — the single biggest compute saver (3-5x)
2. **Playout cap randomization** — free 1.4x speedup
3. **Progressive simulation** — start low (16-50 sims), increase as model improves
4. **Train on all games** — 10-100x more training data vs win-only filtering

## Consequences

**What's easier:**
- Cold-start is eliminated (heuristic bootstrap)
- Training is 5-10x more compute-efficient (Gumbel + playout cap + progressive sims)
- Can go straight to full rules (no curriculum, no transfer gaps)

**What's harder:**
- Must build heuristic player (but it's straightforward for HeXO)
- Gumbel MCTS requires rewriting selection logic (but cleaner than PUCT)
- More moving parts in training pipeline

**What to watch for:**
- Heuristic quality matters for imitation data — if too weak, add 1-2 ply minimax
- Gumbel selection with very few sims may need tuning for 300+ action space
- Progressive simulation schedule needs monitoring (don't increase too fast)

## Applies to

- specs/heuristic-player.md — bootstrap data source
- specs/imitation-learning.md — supervised pre-training
- specs/alphazero-training-v2.md — self-play pipeline (Gumbel, playout cap, progressive)
- alphazero/mcts.py — Gumbel selection replaces PUCT

## References

- KataGo: Accelerating Self-Play Learning in Go (Wu 2019): arxiv.org/abs/1902.10565
- Policy Improvement by Planning with Gumbel (DeepMind 2022): openreview.net/forum?id=bERaNdoegnO
- MiniZero Framework (IEEE ToG 2024): github.com/rlglab/minizero
- Rapfi (GomoCup 2024 champion): github.com/dhbloo/rapfi
- KataGomo (AlphaZero for Gomoku/Connect6): github.com/hzyhhzy/KataGomo
- Warm-Start AlphaZero (2020): arxiv.org/abs/2004.12357
- PCZero path consistency (ICML 2022): github.com/CMACH508/PCZero
