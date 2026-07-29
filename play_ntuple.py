"""runnable file! Plays games with the trained n-tuple agent through the real env.

    python play_ntuple.py --games 500 --budget 0      # greedy policy, instant
    python play_ntuple.py --games 50  --budget 2000   # expectimax, stronger
    python play_ntuple.py --games 1   --budget 50000  # one slow showcase game

Every move goes through envs/SGAI_2048.SGAI_2048, the same environment
rl_play.py uses, so these numbers are comparable to the readme score board. The
fast simulator is only used for training and for the search's own rollouts.
"""

import argparse
import json
import logging
import os
import random
from collections import Counter
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from envs.SGAI_2048 import SGAI_2048, Act2048, logger
from models.NTupleInference import NTupleInference

logger.setLevel(logging.INFO)

DEFAULT_WEIGHTS = os.path.join("models", "ntuple_files", "ntuple_final.npz")
MAX_TURNS_PER_GAME = 100_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument(
        "--budget",
        type=int,
        default=2_000,
        help="expectimax node budget per move; 0 plays the greedy policy",
    )
    parser.add_argument(
        "--max-depth", type=int, default=5, help="ceiling on the search depth"
    )
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--no-downgrade",
        action="store_true",
        help="turn off tile-downgrading, which remaps rare high-tile boards onto "
        "better trained ones before searching",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seeds the tile spawns as well as the agent, so a run repeats exactly",
    )
    parser.add_argument(
        "--record",
        default=None,
        help="write the best game's move by move boards here, for replay_pygame.py",
    )
    parser.add_argument("--plot", action="store_true", help="show the usual plots")
    return parser.parse_args()


def board_to_hex(board) -> str:
    """Pack a board of tile values into 16 nibbles of exponents, as one hex string."""
    packed = 0
    for row_index in range(4):
        for col_index in range(4):
            value = int(board[row_index][col_index])
            exponent = value.bit_length() - 1 if value else 0
            packed |= exponent << (4 * (4 * row_index + col_index))
    return f"{packed:016x}"


def main() -> None:
    args = parse_args()

    # The env spawns tiles with the module level random, so seeding only the
    # agent leaves a run unrepeatable. Seed both, or a game worth keeping
    # cannot be played back.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    env = SGAI_2048(render_mode="rl-headless", num_rows=4, num_cols=4)
    agent = NTupleInference(
        weights_file_path=args.weights,
        node_budget=args.budget,
        max_depth=args.max_depth,
        tile_downgrading=not args.no_downgrade,
        seed=args.seed,
    )

    game_scores: List[int] = []
    game_highest_tiles: List[int] = []
    turn_rewards: List[int] = []
    all_actions: Dict[Act2048, int] = {action: 0 for action in Act2048}

    best_recording = None

    for game_number in range(args.games):
        observation, _ = env.reset()
        steps = [] if args.record else None
        start_board = board_to_hex(env.board)

        for _ in range(MAX_TURNS_PER_GAME):
            action = agent.get_action(obs_raw=observation)
            all_actions[Act2048(action)] += 1
            observation, reward, terminated, truncated, _ = env.step(action)
            turn_rewards.append(reward)
            if steps is not None:
                steps.append(
                    {"a": int(action), "r": int(reward), "b": board_to_hex(env.board)}
                )
            if terminated or truncated:
                break

        if steps is not None:
            recording = {
                "score": int(env.total_score),
                "max_tile": int(env.highest_tile),
                "moves": len(steps),
                "start": start_board,
                "steps": steps,
                "weights": args.weights,
                "budget": args.budget,
                "seed": args.seed,
            }
            # Rank the way the score board does: tile first, score as the tiebreak
            if best_recording is None or (
                recording["max_tile"],
                recording["score"],
            ) > (best_recording["max_tile"], best_recording["score"]):
                best_recording = recording

        game_scores.append(env.total_score)
        game_highest_tiles.append(env.highest_tile)
        print(
            f"game {game_number + 1:>4}/{args.games}  "
            f"score {env.total_score:>7,}  tile {env.highest_tile:>5}  "
            f"(running best: {max(game_scores):,} / {max(game_highest_tiles)})"
        )

    env.close()

    if best_recording is not None:
        directory = os.path.dirname(args.record)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.record, "w", encoding="utf-8") as handle:
            json.dump(best_recording, handle, separators=(",", ":"))
        print(
            f"\nrecorded best game to {args.record} "
            f"({best_recording['score']:,} points, tile {best_recording['max_tile']}, "
            f"{best_recording['moves']:,} moves)"
        )
        print(f"  replay it with: python replay_pygame.py --file {args.record}")

    print("\n--- results ---")
    print(f"games:          {args.games}")
    search_description = (
        "greedy (no search)"
        if args.budget <= 0
        else f"expectimax, budget {args.budget:,}, max depth {args.max_depth}"
    )
    print(f"search:         {search_description}")
    print(f"mean score:     {sum(game_scores) / len(game_scores):,.0f}")
    print(f"best score:     {max(game_scores):,}")
    print(f"highest tile:   {max(game_highest_tiles)}")

    print("\nhighest tile reached, by game:")
    for tile, count in sorted(Counter(game_highest_tiles).items()):
        print(f"  {tile:>5}: {count:>4} games ({count / args.games:6.1%})")

    print("\nfraction of games reaching at least:")
    for tile in (512, 1024, 2048, 4096, 8192):
        reached = sum(1 for value in game_highest_tiles if value >= tile)
        print(f"  {tile:>5}: {reached / args.games:6.1%}")

    best_index = game_scores.index(max(game_scores))
    print(
        f"\nreadme score board line:\n"
        f"| you | {game_highest_tiles[best_index]} | {game_scores[best_index]} | "
        f"n-tuple TD + expectimax | (date) |"
    )

    if args.plot:
        plt.plot(game_scores)
        plt.xlabel("Game")
        plt.ylabel("Final Score (merge points)")
        plt.title("Score per Game")
        plt.show()

        plt.bar(range(len(all_actions)), list(all_actions.values()), align="center")
        plt.xticks(range(len(all_actions)), [str(a) for a in all_actions])
        plt.xlabel("Action")
        plt.ylabel("Times Chosen")
        plt.title("Action Counts")
        plt.show()


if __name__ == "__main__":
    main()
