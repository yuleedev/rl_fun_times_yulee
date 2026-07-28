# BWSI SGAI 2026: 2048 Avalanche

## What this is

A sort of competition. Kind of.

Because of this I, Jo, will be the final arbiter of all things including rules, judging, change management, and everything else.

There will be a few prizes. I left a few questions and things marked as TODOs in the code. That's one path to points. The other path is to post your best model's highest tile and score. You can do so by editing this readme below (using git).

Human scores don't count. This is an agent only competition. RL agents are fair game (and probably the favorite). Human encoded algorithms are good too (possible clever solutions). Mixed methods could be fun. Your agent doesn't necessarily need to be deterministic either. You can be creative.

The game of the game is an agent playing a variant of 2048 that I call 2048 Avalanche. It's extra collapse-y. I like it and it's different enough from the original 2048 game to make things interesting.

Participation is completely voluntary. This is hopefully a fun RL exercise for those with that craving. We understand it's an interesting topic that pairs very well with serious games.

You may self form teams or work individually. You may be on more than one team for all I care. I will work alone. I will also work with any TA who wants to team with me. I won't work with students tho, y'all are on your own ;)

Post your scores below, just have fun playing the human mode of the game, or completely ignore this. Your call!

## Feedback

I expect it to possibly be a bit rough around the edges. Help me out!

If you have thoughts to improve let me or your TA know. What can we do to make this more fun? Did I do something wrong in the code? Whatever type of feedback I'm all ears.

## Contributions via PRs

It's possible I messed up coding something. If there's a major error or any kind of fix you'd like me to address, there are a few approaches. You can reach out and we can talk about it like normal.

Or, you can try out the github workflow called "Pull Requests". You can pull the repo, create a new branch, and then commit any change you want to make. Fix a bug, correct an error, add extra features, etc...

Once your work is committed and ready to be merged, go to the mit github website and browse to the repo. Open the PR tab and follow the directions specifying your branch to be merged with the branch you want to merge it into. You can make notes and start a discussion while providing the code fix. Tag or Assign the PR to me and I will review and merge. It's pretty easy and something I use multiple times a week in my day to day work.

Perhaps the most prolific or critical PR will also get a prize.

# Setup

Create the conda env and activate it:

```
conda env create -f environment.yml
conda activate rl_fun_times
```

Then run any of the runnable files below. Note that pygame comes from pip on purpose (the conda-forge build has a broken SDL dependency on Windows) — the environment.yml already handles this, so don't swap it over to conda's package.

# File breakdown

rl_fun_times :: top level directory for this activity

rl_fun_times -> algos :: Directory containing human written algorithms

- Two given examples are provided :: BadDLOnlyAgent (always down and left) and RandomAgent

rl_fun_times -> envs :: Directory containing the gymnasium compatible 2048 environment, contains all game logic

rl_fun_times -> models :: Directory containing training and inference classes for RL algorithms along with supporting model files in sub directories

- dq_model_files :: directory of saved model weights for a given DQN RL agent
- q_table_files :: directory of saved q table dumps for a given Qlearning RL agent
- DQAgentInference :: Inference agent, loads DQN RL agent weights and gives the trained policy's actions
- DQAgentTrainer :: Training agent, defines and trains a DQN RL agent, contains logic of updating the RL algorithm
- QAgentInference :: Inference agent, loads a q-table dictionary and gives the trained q-table's actions
- QAgentTrainer :: Training agent, defines and trains a q-table dictionary, contains logic of updating the q table

rl_fun_times -> utils :: Directory containing helper functions and general utilities

- logger :: basic logger that is vastly un-utilized. This may be a good starter task for a PR (wink wink)
- state_keys :: symmetry canonicalization for the Q-table — all 8 rotations/mirrors of a board share one table entry. Both Q agents use it so training and inference always agree on keys

rl_fun_times -> top level files :: the following are files directly in the top level directory

- 2048_collapse_logic.pptx :: slides showing a crude visualization of the collapsing algorithms
- algo_play.py :: runnable file! Executes a large number of games using a human algorithm, shows some basic results
- environment.yml :: conda environment spec, see Setup above
- human_play_algo.py :: runnable file! Executes a single game showing the UI, moves are selected by a human algorithm
  - Use w, a, s, d, arrow keys, space, or enter to advance turns
  - q, backspace, escape, or closing the window to quit
- human_play_manual.py :: runnable file! Executes a single game showing the UI, moves are entered by you, the player
  - Use w, a, s, d, arrow keys to collapse in the appropriate direction
  - q, backspace, escape, or closing the window to quit
- human_play_rl.py :: runnable file! Executes a single game showing the UI, moves are selected by an RL algorithm
  - Use w, a, s, d, arrow keys, space, or enter to advance turns
  - q, backspace, escape, or closing the window to quit
- readme.md :: this document :)
- rl_play.py :: runnable file! Executes a large number of games using an RL algorithm, shows some basic results
- rl_training.py :: runnable file! Trains an RL agent, settable hyper parameters and file I/O
- ruff.toml :: lint config. Keeps ruff from stripping the intentionally "unused" imports in rl_training.py (the Q vs DQ toggle)

Peppered around are various \*.log files. I do not handle logging very well which explains this phenomenon. Maybe someone will fix it.

# Run Modes

Human modes: These all load the GUI, variants are provided for each way actions can be selected

- human play -> Algo :: Loads the GUI and the player advances actions determined by a loaded human algorithm
- human play -> Manual :: Loads the GUI and the player advances actions by inputting them directly
- human play -> RL :: Loads the GUI and the player advances actions determined by a loaded RL algorithm

Play Modes: These do NOT load the GUI and runs multiple games at a time, variants provided for human algo and RL agents

- Algo play :: Runs many games with a human algo playing, shows some basic outputs
- RL play :: Runs many games with an RL algo playing, shows some basic outputs

Training Modes: This mode does NOT load the GUI and runs multiple games at a time, no variants

- RL Training :: Sets up and trains an RL agent, saves model periodically during training, basic outputs and plotting

# Score Board

Scoring is classic 2048: every merge is worth the value of the tile it creates, and a move that changes nothing on the board costs 10 points. The game ends when the board is full and nothing can merge. Rankings sort by highest tile reached first, then by score to break ties. The batch play modes (algo_play and rl_play) print both leaderboard numbers at the end of a run. the baseline RL models were trained for about 10 minutes on Jo's computer. final weights are included in the models/ dir.
NOTE: do not submit human play scores, my human play score is there only as a quality reference for the baseline RL / Algo scores.

| Name/Initials/Team Name | Highest Tile | Score | Method / Algo Type | Date    |
| ----------------------- | ------------ | ----- | ------------------ | ------- |
| Baseline                | 256          | 3314  | down left greedy   | 20 July |
| Baseline                | 512          | 3662  | Q Learning         | 20 July |
| Baseline                | 256          | 2396  | Deep Q Learning    | 20 July |
| Jo                      | 512          | 7522  | Human Play         | 20 July |
| Jo                      | 1024         | 14088 | Deep Q Learning    | 22 July | 
| Max                     | 2048         | 31264 | double-DQN / masks | 24 July |
|                         |              |       |                    |         |
|                         |              |       |                    |         |
|                         |              |       |                    |         |
|                         |              |       |                    |         |
|                         |              |       |                    |         |
|                         |              |       |                    |         |
|                         |              |       |                    |         |

bonus points if you beat me with deep q learning. i spent 2 days tuning and training. if you do better, let me know asap! :)

FINAL CHALLENGE! 
Big points and plushies for any good algorithms or methods for determining if a score is real or fake (or possible)
