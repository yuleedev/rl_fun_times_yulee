# 2048 Avalanche RL

2048 Avalanche RL is my entry for the reinforcement learning challenge in the Serious Games AI course at MIT Beaver Works Summer Institute (BWSI) 2026. The challenge was to build an agent that gets the highest tile and score it can in "2048 Avalanche," a harder version of 2048. The starter code, game environment, and baseline agents came from the course instructor. I added the n-tuple network agent and the expectimax search.

## Result

My best agent reached the **16384 tile** with a score of **255,632**. That game is saved in `replays/attempt_30.json`, so you can watch it! Iirc my score was the second or third highest amongst the other competitiors in my cohort :)

| Method | Highest Tile | Score |
| --- | --- | --- |
| Down-left greedy (baseline) | 256 | 3,314 |
| Q-learning (baseline) | 512 | 3,662 |
| Deep Q-learning (baseline) | 256 | 2,396 |
| **N-tuple network + expectimax (mine)** | **16384** | **255,632** |

## How 2048 Avalanche Is Different

- Merges chain together, so `[2, 2, 4]` collapses all the way into `[8]`.
- New tiles are 2 or 4 at a 50/50 chance, not the usual 90/10.
- A new tile appears every turn, even when your move didn't change the board.
- A move that changes nothing costs 10 points.

## How the Agent Works

- **N-tuple network:** Scores a board by looking up small groups of 6 cells in large weight tables. All 8 rotations and mirrors of a board share the same tables.
- **TD(0) training:** The network learns by playing thousands of games on its own. It uses afterstates and temporal coherence, which tunes the learning rate for each weight on its own.
- **Fast simulator:** The board is packed into one 64-bit integer so training and search run quickly. `verify_fast_env.py` checks that it matches the real game exactly.
- **Expectimax search:** At play time, the agent looks a few moves ahead and averages over where the next tile could appear. It picks the search depth from a node budget, so each move takes about the same time.

## Getting Started/Installation

Create and activate the conda environment:

    conda env create -f environment.yml
    conda activate rl_fun_times

The trained weights are already included in `models/ntuple_files/ntuple_final.npz`.

## How to Run

**Watch the record game:**

    python replay_pygame.py --file replays/attempt_30.json

**Play games with the trained agent:**

    python play_ntuple.py --games 50 --budget 2000

**Train the agent from scratch:**

    python train_ntuple.py --games 100000

**Play the game yourself:**

    python human_play_manual.py

## Controls

- **WASD / Arrow keys:** Slide the tiles in manual play.
- **Space:** Play or pause the replay viewer.
- **Q / Esc:** Quit.