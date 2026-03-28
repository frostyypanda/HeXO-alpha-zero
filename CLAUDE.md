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
