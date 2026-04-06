# Feature: Heuristic Player

## Status
draft

## Why this exists

AlphaZero training requires initial signal — the NN must produce games with winners for
the value head to learn from. Our prior attempts showed that:

- Random play at win>=5 / dist>=4 produces <1% wins (not enough signal)
- MCTS with an untrained NN produces FEWER wins than random play
- Curriculum from simpler rules doesn't transfer to harder rules

A heuristic player solves this by generating unlimited winning games at any difficulty
level. It doesn't need to be optimal — just good enough to:
1. Consistently win games (>90% of games end decisively)
2. Play "reasonable" moves that the NN can learn from
3. Work at full rules (win=5, dist=8) from day one

## What it does

### Position Evaluation

Scores each candidate move by analyzing tactical patterns across the 3 hex axes
(horizontal, diagonal-left, diagonal-right). Two components:

**Offensive score** — how much does this move advance my own lines?
- Count my consecutive stones in each direction through this cell
- Exponential scaling: longer lines score exponentially more
- Open-end bonus: lines with both ends open are more threatening
- A move that creates length win_length-1 with an open end is nearly winning

**Defensive score** — how urgently does the opponent need to be blocked?
- Mirror of offensive scoring, applied to opponent's lines through empty cells
- Immediate threats (opponent at win_length-1) get maximum priority
- Near-threats (win_length-2 with open ends) get high priority

**Combined score** for move m:
```
score(m) = offense(m) + defense_weight * defense(m) + noise
```

Where:
- `defense_weight` controls how reactive vs proactive the player is (~0.9)
- `noise` adds controlled randomness for diverse training data

### Move Selection

Given scores for all legal moves:
1. Compute softmax probabilities: `p(m) = exp(score(m) / temperature) / Z`
2. Sample from this distribution (temperature controls diversity)
3. High temperature (~1.5-2.0) for diverse training data
4. Low temperature (~0.3) for strong play / evaluation

### HeXO-Specific Tactical Awareness

The 2-moves-per-turn structure creates unique tactical patterns:

- **Fork creation**: Place stone A to create threat in direction X, stone B to create
  threat in direction Y. Opponent can only block one.
- **Double extension**: Both stones extend the same line, jumping ahead 2 cells per turn.
- **Threat + block**: Use one stone to block opponent, other to advance own line.

The heuristic doesn't need to plan these explicitly — scoring individual moves by line
potential naturally produces this behavior because the highest-scoring moves tend to be
the ones that create or extend threats.

### Strength Levels

Parameterized for generating diverse training data:

| Level | Temperature | Defense Weight | Use Case |
|-------|-------------|----------------|----------|
| Strong | 0.3 | 0.95 | Evaluation opponent, late-game training data |
| Medium | 1.0 | 0.85 | Bulk training data generation |
| Weak | 2.0 | 0.70 | Easy opponent for early bootstrapping |

## Boundaries & edge cases

**What it intentionally does NOT do:**
- No tree search / lookahead — evaluates positions one move at a time
- No opening book — relies on general heuristic from move 1
- No endgame solving — doesn't compute forced wins, just scores threats
- No awareness of the 2-move-per-turn structure at the planning level (but the game
  engine handles turn structure; the heuristic just scores individual stone placements)

**Known limitations:**
- Will miss deep tactical sequences that require multi-move planning
- Can be exploited by an opponent that plans forks 3+ moves ahead
- Defensive scoring is reactive (blocks existing threats, doesn't prevent future ones)

**These limitations are acceptable** because the heuristic exists only to bootstrap the
NN. It doesn't need to play optimally — it needs to play well enough that its games
contain useful positional patterns for imitation learning.

## Testing & verification

### Key scenarios
- [ ] Heuristic player wins >90% of games against random player at win=5/dist=8
- [ ] Heuristic vs heuristic games end in wins (not draws) >80% of the time
- [ ] Heuristic player builds lines (avg longest line per game >= 3)
- [ ] Heuristic blocks opponent threats of length win_length-2 or higher
- [ ] Move scoring produces distinct preferences (not uniform) for typical positions

### Edge cases
- First move (no existing stones): should place near origin — expected: any valid move
- Very spread-out boards (dist=8): should still score near existing clusters — expected:
  heuristic prefers moves near its own stones
- Opponent one move from winning: should block — expected: defensive score dominates

### Automation notes
- Unit tests for scoring function (known positions with expected relative scores)
- Integration test: 100 games heuristic vs random, check win rate
- Integration test: 100 games heuristic vs heuristic, check decisive rate

## Decision log
| Date | Decision | Why | Who |
|------|----------|-----|-----|
| 2026-04-06 | Single-move heuristic, no tree search | Simplicity; purpose is data generation not optimal play. Tree search is what MCTS/NN will handle. | Kei + Claude |
| 2026-04-06 | Exponential line scoring | Linear scoring undervalues long lines; a 4-in-a-row is much more than 2x a 2-in-a-row | Kei + Claude |
| 2026-04-06 | Parameterized strength levels | Need diverse training data: strong games for policy quality, weaker games for coverage | Kei + Claude |

## Related
- [specs/_overview.md](_overview.md) — epic context
- [specs/imitation-learning.md](imitation-learning.md) — consumes heuristic game data
- `hexo/hex_utils.py` — line counting utilities (count_in_direction, check_line_through)
- `hexo/game.py` — game state and move generation
