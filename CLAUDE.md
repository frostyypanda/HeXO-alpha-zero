# HeXO AlphaZero

## Project Goal
Train a local AI to play **HeXO** (Infinity Hexagonal Tic-Tac-Toe) using the AlphaZero self-play reinforcement learning approach. Based on an existing Colab notebook that implements AlphaZero for standard Tic-Tac-Toe and Connect Four: https://colab.research.google.com/drive/1Z70Rt7NWB8gaumV_RZhqOmKoqJosBc8T

## HeXO Game Rules

**Source:** https://hexo.did.science/rules | **Repo:** https://github.com/WolverinDEV/infhex-tic-tac-toe

- **Board:** Infinite hexagonal grid (3 axes: horizontal + 2 diagonals).
- **Turn order:**
  - Player 1 places **1 hex** (opening move).
  - Player 2 places **2 hexes**.
  - All subsequent turns: **2 hexes** per player.
- **Win condition:** First to align **N of their own hexes** in a straight line along any of the 3 axes.
  - Official HeXO: N=6. Training default: N=5 (configurable via `--win-length`).
  - Curriculum: start with lower N for faster training signal, then increase.
- **Placement rule:** A new hex must be placed **at most 8 cells apart** from any existing hex (prevents scattered play).
- **No fixed board size** — the board expands dynamically as players place hexes.

## Key Technical Challenges

### 1. Game Implementation (Python)
Write the HeXO game logic:
- Hex grid coordinate system (axial coordinates `(q, r)` recommended — the third axis `s = -q - r` is implicit).
- Move generation: all empty hexes within distance 8 of any existing hex.
- Win detection: check all 3 axes for 6-in-a-row after each placement.
- Handle the asymmetric turn structure (1 / 2 / 2 / 2 / ...).
- Game state must conform to whatever abstract `Game` interface the AlphaZero framework expects (from the Colab notebook).

### 2. Board Representation for the Neural Network
**Approach: Dynamic Bounding Box + Fully Convolutional Network (FCN).**

The infinite board is encoded as the tight bounding box of all placed pieces, padded
by `max_placement_dist` (8) cells in every direction. This guarantees:
- **0% piece loss** — every placed stone is always inside the grid, by construction.
- **All legal moves are in-bounds** — the +8 padding covers the placement rule exactly.
- **All empty gaps between pieces are visible** — the CNN sees the full spatial layout.
- **Grid grows organically** — small early game (~17x17), larger only when pieces spread.

Encoding: 3-channel `(3, H, W)` float32 tensor where H and W vary per state:
  - Channel 0: current player's stones
  - Channel 1: opponent's stones
  - Channel 2: empty cells (1 where empty, 0 where occupied)

For batched inference (parallel self-play), states are padded to the max (H, W) in
the batch with zeros, and padding cells are masked as illegal in the policy output.

**Why not a fixed window?** A fixed window (e.g. 21x21 or 51x51) risks losing track
of pieces if the game spreads beyond the window. During self-play training the AI
makes unpredictable moves, and an adversary could deliberately spread to exploit blind
spots. The dynamic bounding box eliminates this risk entirely.

**Why not GNN/Transformer?** GNNs require many layers for global propagation and add
PyTorch Geometric complexity. Transformers scale O(N^2). The FCN approach is proven
(AlphaZero for Go), simple, and fast on GPU.

### 3. Neural Network Architecture
The NN must be **fully convolutional** (no FC layers in the body) to accept the
variable-size grid input.

- **Body:** ResNet-style residual blocks: Conv2d → BatchNorm2d → ReLU, with skip
  connections. No fully connected layers. Accepts any (3, H, W) input.
- **Policy head:** Conv2d → (1, H, W) → flatten → H*W action logits. Automatically
  scales with the grid. Illegal moves are masked before softmax.
- **Value head:** GlobalAvgPool (kills spatial dims) → FC → tanh → scalar in [-1, 1].
  Always produces a fixed-size output regardless of input grid dimensions.
- Start with 5-10 res blocks, 128 filters. Scale up given the 4090's capability.

### 4. MCTS + Search Depth / Time Control
- AlphaZero uses **MCTS (Monte Carlo Tree Search)** guided by the neural network (no random rollouts).
- **Configurable search budget:**
  - By **simulation count** (e.g., 800 simulations per move — controls how many moves ahead the tree effectively explores).
  - By **time limit in milliseconds** — run as many simulations as possible within the budget. This is the preferred mode for play, as it naturally adapts to position complexity.
- During **training**, use a fixed simulation count for reproducibility.
- During **play/inference**, use a time limit (ms) for practical real-time play.
- Temperature parameter for move selection: high early in the game (exploration), low/zero later (exploitation).

## Training Setup
- **Hardware:** AMD 7800X3D (16 threads), 32 GB RAM, NVIDIA RTX 4090 (24 GB VRAM).
- **Framework:** PyTorch with CUDA.
- Self-play loop: generate games via MCTS + current net -> train net on (state, policy, value) tuples -> repeat.
- Store training data in a replay buffer.
- Periodically pit the new model against the previous best; keep the stronger one.

### Parallelization Strategy
Training must be parallelized to fully utilize the hardware:
- **Parallel self-play:** Run multiple self-play games concurrently using Python `multiprocessing` (CPU workers). Each worker runs MCTS with its own game state. The 7800X3D has 16 threads — target ~8-12 parallel self-play workers to leave headroom for the system and training.
- **Batched neural network inference:** Collect pending MCTS leaf evaluations from all workers into a batch and run a single forward pass on the GPU. This is critical — individual inference calls waste GPU throughput. Use a shared queue or `torch.multiprocessing` with a dedicated GPU inference server process.
- **Async pipeline:** Self-play and training can overlap: workers generate data into a shared replay buffer while a separate process trains the network on recent data. The training process periodically publishes updated weights that self-play workers pick up.
- **GPU training:** Standard PyTorch DataLoader with pinned memory and multiple workers for the training loop itself. The 4090 can handle large batch sizes (512-2048+).
- **Optional: parallel MCTS within a single game** — run multiple simulation paths in parallel using virtual losses (mark in-flight nodes as losses to encourage exploration of different branches).

## Project Phases

### Phase 1: Game Engine
- Implement HeXO game logic in Python with full rule support.
- Unit tests for move generation, win detection, turn structure.

### Phase 2: AlphaZero Framework Adaptation
- Port/adapt the Colab notebook's AlphaZero framework to work with HeXO.
- Implement dynamic bounding box encoding (variable-size grid, 0% piece loss).
- Build fully convolutional ResNet (no FC layers in body) for variable-size input.
- Adapt MCTS to handle variable action space sizes per state.

### Phase 3: Training Pipeline
- Self-play data generation (parallelized on GPU).
- Training loop with configurable hyperparameters.
- Model checkpointing and evaluation.

### Phase 4: Play Interface
- Allow human vs AI play with configurable time limits.
- Visualize the hex board and AI's move probabilities.

## File Structure (Planned)
```
HeXO-alpha-zero/
  CLAUDE.md          # This file
  hexo/
    game.py          # HeXO game logic, rules, state
    hex_utils.py     # Hex grid utilities, coordinate math
  alphazero/
    mcts.py          # Monte Carlo Tree Search (supports virtual loss for parallel sims)
    model.py         # Neural network (ResNet dual-head)
    train.py         # Training loop (reads from replay buffer, writes new weights)
    self_play.py     # Parallel self-play game generation (multiprocessing workers)
    inference.py     # Batched GPU inference server for MCTS evaluations
    arena.py         # Model evaluation (pit models against each other)
    pipeline.py      # Orchestrator: launches self-play workers, inference server, and trainer
  tests/
    test_game.py     # Game logic tests
    test_mcts.py     # MCTS tests
  play.py            # Human vs AI interface
  config.py          # Hyperparameters, search budget, window size
  requirements.txt
```

---

# Claude Behaviour Rules

## Spec-Anchored Development

Every new feature or behavior change **must** start with a spec.

### Philosophy

A spec is not documentation about code. A spec is upstream of code — the source of truth for intent. Code implements the spec, not the other way around. When spec and code disagree, the spec needs a conscious decision — not a silent override.

The key difference from "spec-first" (where the spec dies after implementation) is that the spec lives as long as the feature lives, and is updated alongside code for the entire lifetime of that feature.

| Level | Spec lives... | Human edits... |
|-------|---------------|----------------|
| Spec-first | until task is done | code |
| Spec-anchored | while feature lives | spec + code |
| Spec-as-source | forever | only spec, code is generated |

### The 5 Rules

1. **Spec lives in the repo next to code** — not in Confluence, not in a ticket.
2. **Every behavior change = update spec first** — if you change behavior, the spec changes in the same commit/PR.
3. **Spec describes INTENT, not implementation** — what the feature does, why, boundaries, edge cases. Not how (that's the code).
4. **Code review includes spec review** — if a PR changed behavior but the spec wasn't updated, that's a bug in the review process.
5. **AI agents read the spec before making changes** — the agent gets the spec as context, works within those boundaries, and updates the spec when done.

### File Naming

- One file per standalone feature: `feature-name.md`
- Kebab-case: `calendar-preferences.md`, `payment-history.md`
- Short but descriptive

### Standalone vs Epic

- Feature is self-contained, one file covers it → flat spec
- Product initiative has 3+ distinct implementable features sharing motivation → epic directory

### Epic Directory Convention

- `_overview.md` — epic-level context, links to origin PRD, high-level vision. Not a spec itself.
- Child specs — each follows the standard template, self-contained with own status/boundaries/testing/decision log. References `_overview.md` for broader context.

### Spec Template

```markdown
# Feature: [Name]

## Status
draft | approved | implemented | modified | deprecated

## Why this exists
Who asked for it, what pain it solves, business motivation.
Support can read this. Marketing can read this.

## What it does
Behavior description — what the user experiences.
Not code, not implementation. Just: "when X happens, user sees Y."

## Boundaries & edge cases
What it intentionally does NOT do. Known limitations.
This is gold for AI agents — "stay inside these lines."

## Testing & verification
How we know this works. What to automate, what to smoke-test.

### Key scenarios
- [ ] Scenario 1
- [ ] Scenario 2

### Edge cases
- Edge case 1 — expected behavior
- Edge case 2 — expected behavior

### Automation notes
What's worth automating vs manual spot-check.

## Decision log
| Date | Decision | Why | Who |
|------|----------|-----|-----|
| YYYY-MM-DD | What was decided | Reasoning | Who decided |

## Related
- Links to related specs, code paths, or external references
```

### Cross-Cutting Technical Decisions

For decisions that span multiple features or establish reusable patterns:

Naming: `NNN-short-descriptive-name.md` (numbered)

Template:

```markdown
# Decision NNN: Title

## Status
accepted | superseded | deprecated

## Context
What situation or problem prompted this decision.

## Options considered
1. **Option A** — description. Pros/cons.
2. **Option B** — description. Pros/cons.

## Decision
What we chose and why.

## Consequences
What follows — what's easier, what's harder, what to watch for.

## Applies to
Which specs, features, or areas of the codebase this affects.
```

### Rules for AI Agents

When working on a feature with an existing spec:
1. Read the spec first — before touching any code
2. Work within the boundaries — the spec defines what the feature should do
3. Use testing scenarios as acceptance criteria
4. Update the spec when done — if you changed behavior, the spec must reflect it
5. Update the decision log — if you made a non-obvious choice, log it with reasoning

When creating a new feature:
1. Create the spec first — before writing code
2. Get approval on the spec — align on intent before implementation
3. Implement within spec boundaries — the spec is the contract
4. Fill in testing scenarios — as you discover edge cases during implementation

Mandatory highlights:
- **Before writing code** — create or update the spec for the feature being changed.
- **Spec and code change together** — spec updates ship in the same commit/PR as code changes.
- **Spec describes intent, not implementation** — what, why, boundaries, edge cases. Code handles the how.
- **AI agents**: before modifying a feature, check for an existing spec. Read it first. Update it when done. Remind the user if a behavior change doesn't have a corresponding spec update.
- **Cross-cutting technical decisions** should be recorded. Check existing decisions before making architectural choices that span multiple features.

---

## Code Size & Complexity Limits

Small functions and shallow nesting are non-negotiable. If a function is hard to read, it's wrong — split it.

### Function/Method Length

| Language | Max lines per function | Hard ceiling |
|----------|----------------------|--------------|
| Python | 20 lines | 30 lines (requires justification in PR) |
| TypeScript | 20 lines | 30 lines |

Lines = logical lines, not counting blank lines, comments, or type annotations. If you're at 20, look for an extract. If you're at 30, you've already gone too far.

### Nesting Depth

| Language | Max indentation levels | Meaning |
|----------|----------------------|---------|
| Python | 3 levels | function → if → for (stop here) |
| TypeScript | 3 levels | function → if → for (stop here) |

If you need a 4th level, extract the inner block into a named function. Early returns, guard clauses, and `continue` are your friends:

```python
# BAD — 4 levels deep
def process_orders(orders):
    for order in orders:
        if order.is_valid:
            if order.side == "buy":
                if order.price > 0:
                    execute(order)

# GOOD — flat
def process_orders(orders):
    for order in orders:
        if not order.is_valid:
            continue
        if order.side != "buy":
            continue
        if order.price <= 0:
            continue
        execute(order)
```

### File Size

| Language | Target max lines | Hard ceiling |
|----------|-----------------|--------------|
| Python | 200 lines | 300 lines |
| TypeScript/TSX | 200 lines | 300 lines |
| Test files | 400 lines | 600 lines |

If a file hits 200 lines, look for a natural split. If it hits 300, split it before adding more code.

### Enforcement

- **Code review**: reject PRs that violate these limits without justification
- **AI agents**: when writing code, check function length and nesting before finishing. Split proactively — don't wait for review.
- **Exceptions**: data class definitions, configuration blocks, and auto-generated code are exempt from line counts

---

## Test Writing Rules

### Core Philosophy

Based on [Testing Overview](https://abseil.io/resources/swe-book/html/ch12.html) and [Test Doubles](https://abseil.io/resources/swe-book/html/ch13.html).

### Rule 1: Test Behaviors, Not Methods

A test should verify what the code does for the user, not mirror internal structure. One method may need multiple tests (multiple behaviors). Multiple methods may need one test (one behavior spanning them).

```
// BAD — one test per method, testing internals
test("calculateTotal()")
test("applyDiscount()")
test("formatPrice()")

// GOOD — one test per behavior the user cares about
test("order with coupon shows discounted total at checkout")
test("order without coupon shows full price")
test("expired coupon is rejected with clear message")
```

### Rule 2: Test State, Not Interactions

Assert on outcomes (return values, database state, UI output), not on whether specific internal functions were called or in what order.

Exception: interaction testing is acceptable when the side effect IS the behavior (e.g., verifying an external API was called in an integration test).

### Rule 3: DAMP > DRY

Tests should be Descriptive And Meaningful Phrases. Repeating setup code across tests is fine if it makes each test self-contained and readable. A reader should understand a test without jumping to shared helpers.

### Rule 4: No Logic in Tests

No `if`, `else`, loops, ternaries, string concatenation, or conditional assertions in test bodies. If you need branching, write separate tests.

### Rule 5: Given / When / Then Structure

Every test has three clearly separated phases:
1. **Given** — set up the preconditions
2. **When** — perform the single action under test
3. **Then** — assert on outcomes

### Rule 6: Make Tests Complete and Concise

- Include all relevant setup in the test body (complete)
- Only set fields that affect the behavior under test (concise)
- Use factory defaults for everything else

### Rule 7: Test Names Should Be Sentences

A test name should describe the behavior, not the method. Pattern: "[unit] [does something] [when/given condition]".

### Rule 8: Prefer Real Implementations Over Mocks

Use real implementations wherever practical. Only use test doubles when:
- The real implementation is slow (network, disk, heavy computation)
- The real implementation is nondeterministic (time, randomness, external services)
- The real implementation is unavailable in the test environment

When you must mock, prefer fakes > stubs > mocks.

### Rule 9: Don't Overuse Mocks

Signs you're over-mocking:
- Mock setup is longer than the test itself
- You're mocking things you own
- You're asserting on mock call arguments rather than outcomes

### Rule 10: Tests Must Be Deterministic

No dependency on wall-clock time, execution order, external services, random data, or shared mutable state between tests.

### Rule 11: Make Tests Hermetic

Each test sets up everything it needs, runs in isolation, and cleans up after itself. Must produce the same result whether run alone, in a suite, or in parallel.

### Rule 12: Tests Should Be Fast

- Unit tests: < 2 seconds per test
- Never `sleep()` in a test — use polling with short intervals or event-driven waits
- Slow tests should be classified differently (integration) and run in a separate suite

### Rule 13: Unchanging Tests

The ideal test is written once and never changed unless the behavior it covers changes. If refactoring internals breaks your tests, they're testing implementation, not behavior.

### Rule 14: One Assertion Per Behavior (Not Per Test)

Multiple asserts in one test are fine when they verify different facets of one behavior. Multiple asserts verifying different behaviors should be separate tests.

### Rule 15: Test Public APIs, Not Internal Details

Test through the same interface your callers use. If a function is private/internal, test it through the public function that calls it.

### Test Types

| Type | What it tests | Dependencies | Speed |
|------|---------------|--------------|-------|
| Unit | Business logic in isolation | All external deps mocked/faked | < 2s per test |
| Integration | Components working together, real I/O | Real database, file system, etc. | 2-30s per test |
| UI / Component | Rendering, interaction, accessibility | DOM environment, usually mocked backend | Medium |
| End-to-end | Full user workflows through the real system | Everything real | Slow |

Sizing guidance: ~80% unit, ~15% integration, ~5% end-to-end.

### Test Data

1. Use factories — never construct test objects by hand with all fields
2. Override only relevant fields — let factory defaults handle the rest
3. Use deterministic IDs — predictable test data is easier to debug
4. Clean up after tests — use shared cleanup utilities

### Hard Rules

1. Never commit with failing tests — all tests pass before merge
2. No feature is complete without tests
3. Never dismiss failures as "pre-existing" — fix or escalate
4. No arbitrary timeouts — fix root causes of slowness
5. No `eslint-disable` / `lint:ignore` — fix the issue or split the test
6. Keep test files short (under 600 lines) — split into focused files
7. Don't duplicate test utilities — use shared helpers
8. Run fast checks first — lint → typecheck → unit → integration

### References

- https://abseil.io/resources/swe-book/html/ch11.html
- https://abseil.io/resources/swe-book/html/ch12.html
- https://abseil.io/resources/swe-book/html/ch13.html
- https://abseil.io/resources/swe-book/html/ch14.html

---

## Experimentation & Scientific Method

Every investigation is an experiment. Treat it like science, not hacking.

### 1. Before You Start

- Check what's already known. Search existing findings before running anything.
- State a hypothesis. "I think X will happen because Y." If you can't state a hypothesis, you don't understand the problem yet.
- One variable at a time. If you change two things and it works, you don't know which one fixed it.

### 2. Structure

- Every experiment gets its own directory
- Every experiment gets a `README.md` with: hypothesis, setup, procedure, raw results, conclusion
- Every experiment gets an entry in an index with a one-line finding summary
- If it modifies production code, it MUST be on a dedicated branch

### 3. During the Experiment

- Document everything. Exact commands, raw output, timestamps.
- Document failures. They're MORE valuable than successes — they eliminate paths.
- Reproduce before concluding. One success is an anecdote. Three successes is evidence.
- Don't fix mid-experiment. If something breaks, document the failure, then start a new attempt with the fix.

### 4. After the Experiment

- Extract durable findings into a findings document
- Distinguish what you built from what you observed. Code design = spec. Hardware/library behavior = finding. Choice between alternatives = decision.
- Update the index with key finding.
- Don't merge experiment branches until findings are proven correct and code is ready for production.

### 5. Audits (Reviewing Past Work)

- Re-read findings with fresh eyes. Are they still accurate?
- Check if experiment conclusions made it into specs or code. If not, why?
- Flag stale findings — mark them with a date and "needs re-validation" if the hardware/software has changed since.
