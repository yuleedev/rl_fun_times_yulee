"""runnable file! Proves utils/bitboard.py behaves exactly like envs/SGAI_2048.py.

The n-tuple agent trains against the bitboard simulator, not the real env, which
is only safe if the two implement the same game. This checks:

  1. Slides. For thousands of random boards (including ones stuffed with
     duplicate tiles, where the cascading merges bite), drive the env's own
     _slide_* methods and compare the board and merge score against
     bitboard.slide.
  2. Game over. Compare bitboard.is_game_over against the env's _is_game_over on
     the same boards.
  3. Whole games. Play random games through the real env and check every step:
     the reward matches, and the board the env returns is our afterstate plus a
     single spawned 2 or 4.

Run it after any change to either file. Prints "ALL CHECKS PASSED" or raises.
"""

import random

import numpy as np

from envs.SGAI_2048 import SGAI_2048
from utils import bitboard as bb

SLIDE_TRIALS = 20_000
GAME_TRIALS = 300


def random_board(rng: random.Random, duplicate_heavy: bool) -> np.ndarray:
    """Build a random 4x4 board of tile values (0 means empty)."""
    if duplicate_heavy:
        # Tiny alphabet, so adjacent equal tiles (and merge cascades) happen
        # constantly.
        alphabet = [0, 2, 2, 4, 4]
    else:
        alphabet = [0] + [1 << e for e in range(1, 12)]
    return np.array(
        [[rng.choice(alphabet) for _ in range(4)] for _ in range(4)], dtype=int
    )


def check_slides(rng: random.Random) -> None:
    env = SGAI_2048(render_mode="rl-headless")
    slide_methods = {
        bb.UP: env._slide_up,
        bb.DOWN: env._slide_down,
        bb.LEFT: env._slide_left,
        bb.RIGHT: env._slide_right,
    }
    for trial in range(SLIDE_TRIALS):
        board_values = random_board(rng, duplicate_heavy=trial % 2 == 0)
        packed = bb.pack(board_values)
        for action, slide_method in slide_methods.items():
            env.board = board_values.copy()
            env.step_merge_score = 0
            slide_method()
            expected_board = env.board
            expected_score = env.step_merge_score

            actual_packed, actual_score = bb.slide(packed, action)
            actual_board = np.array(bb.unpack(actual_packed), dtype=int)

            if not np.array_equal(expected_board, actual_board):
                raise AssertionError(
                    f"slide mismatch on action {action}\n"
                    f"input:\n{board_values}\n"
                    f"env:\n{expected_board}\n"
                    f"bitboard:\n{actual_board}"
                )
            if expected_score != actual_score:
                raise AssertionError(
                    f"score mismatch on action {action}: env {expected_score} "
                    f"vs bitboard {actual_score}\ninput:\n{board_values}"
                )
    print(f"  slides: {SLIDE_TRIALS * 4:,} comparisons OK")


def check_game_over(rng: random.Random) -> None:
    env = SGAI_2048(render_mode="rl-headless")
    checked_over = 0
    for trial in range(SLIDE_TRIALS):
        # Mostly-full boards, so the terminal case actually shows up
        board_values = random_board(rng, duplicate_heavy=trial % 3 == 0)
        if trial % 2 == 0:
            board_values[board_values == 0] = 2 << (trial % 5)
        env.board = board_values.copy()
        env._is_game_over()
        actual = bb.is_game_over(bb.pack(board_values))
        if env.game_over != actual:
            raise AssertionError(
                f"game over mismatch: env {env.game_over} vs bitboard {actual}"
                f"\n{board_values}"
            )
        checked_over += int(actual)

        # A board is terminal exactly when nothing can move it
        packed = bb.pack(board_values)
        if bool(bb.legal_actions(packed)) == actual:
            raise AssertionError(f"legal_actions disagrees with is_game_over\n{board_values}")
    print(f"  game over: {SLIDE_TRIALS:,} boards OK ({checked_over:,} terminal)")


def check_full_games(rng: random.Random) -> None:
    env = SGAI_2048(render_mode="rl-headless")
    total_steps = 0
    for _ in range(GAME_TRIALS):
        observation, _ = env.reset()
        packed = bb.pack(observation)
        if bin(packed).count("1") == 0:
            raise AssertionError("reset produced an empty board")

        while True:
            action = rng.choice(bb.ACTIONS)
            afterstate, merge_score = bb.slide(packed, action)
            changed = afterstate != packed
            expected_reward = merge_score if changed else bb.NO_MOVE_PENALTY

            observation, reward, terminated, _, _ = env.step(action)
            total_steps += 1

            if reward != expected_reward:
                raise AssertionError(
                    f"reward mismatch: env {reward} vs bitboard {expected_reward}"
                )

            # The env spawns a tile after the slide, so everything else should
            # be untouched and the one changed cell should hold a 2 or a 4.
            actual_packed = bb.pack(observation)
            differences = [
                cell
                for cell in range(bb.NUM_CELLS)
                if (actual_packed >> (4 * cell)) & 0xF
                != (afterstate >> (4 * cell)) & 0xF
            ]
            had_room = bool(bb.empty_cells(afterstate))
            if had_room:
                if len(differences) != 1:
                    raise AssertionError(
                        f"expected exactly one spawned tile, got {len(differences)}"
                        f"\nafterstate:\n{bb.render(afterstate)}"
                        f"\nenv:\n{observation}"
                    )
                spawned = (actual_packed >> (4 * differences[0])) & 0xF
                if spawned not in (1, 2):
                    raise AssertionError(f"spawned tile was 2^{spawned}, expected 2 or 4")
            elif differences:
                raise AssertionError("env changed a full afterstate")

            if terminated != bb.is_game_over(actual_packed):
                raise AssertionError("terminal flag mismatch")

            packed = actual_packed
            if terminated:
                break
    print(f"  full games: {GAME_TRIALS} games / {total_steps:,} steps OK")


def main() -> None:
    rng = random.Random(20260728)
    print("Verifying utils/bitboard.py against envs/SGAI_2048.py")
    check_slides(rng)
    check_game_over(rng)
    check_full_games(rng)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
