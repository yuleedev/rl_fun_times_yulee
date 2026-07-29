"""Fast re-implementation of the 2048 Avalanche board mechanics.

SGAI_2048.step loops over every cell in Python and copies the board each turn,
which is far too slow for n-tuple training (tens of millions of moves) or
expectimax search. Here the whole 4x4 board is one 64-bit int: cell
i = 4 * row + col occupies bits 4*i .. 4*i+3 and holds the tile's exponent
(0 empty, 1 = "2", 11 = "2048"), so sliding a row is one lookup in a
precomputed 65536-entry table.

verify_fast_env.py runs this and SGAI_2048 in lockstep to confirm they agree.

Avalanche rules reproduced here: merges cascade, so [2,2,4] -> [8]; a new tile
is 2 or 4 at 50/50 rather than the usual 90/10; and a tile spawns every turn
there is room, even after a move that changed nothing.
"""

import random
from typing import List, Tuple

# Act2048 values, duplicated as plain ints to keep this module import-light
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = (UP, DOWN, LEFT, RIGHT)

NUM_ROWS = 4
NUM_COLS = 4
NUM_CELLS = NUM_ROWS * NUM_COLS
MAX_EXPONENT = 15  # a nibble holds 0..15, i.e. up to the 32768 tile

NO_MOVE_PENALTY = -10  # must match SGAI_2048.no_move_penalty


def _slide_row_left(row: List[int]) -> Tuple[List[int], int]:
    """Collapse one row toward index 0, in exponent space.

    Transcribed from SGAI_2048._slide_left. Two details make this Avalanche
    rather than classic 2048: the inner loop keeps walking left after a merge,
    and a mismatch continues instead of breaking, so one slide can chain
    several merges into a single tile.
    """
    row = list(row)
    score = 0
    for col_main in range(1, NUM_COLS):
        if row[col_main] == 0:
            continue
        for col_inner in range(col_main - 1, -1, -1):
            if row[col_inner] == 0:
                row[col_inner] = row[col_inner + 1]
                row[col_inner + 1] = 0
            elif row[col_inner] == row[col_inner + 1]:
                merged = row[col_inner] + 1
                score += 1 << merged  # merges score the new tile's value
                # A nibble caps at 15 (the 32768 tile), which a real game can
                # never hit. The clamp is only here because the table builder
                # enumerates every bit pattern.
                row[col_inner] = min(merged, MAX_EXPONENT)
                row[col_inner + 1] = 0
            else:
                continue
    return row, score


def _pack_row(row: List[int]) -> int:
    return row[0] | (row[1] << 4) | (row[2] << 8) | (row[3] << 12)


def _unpack_row(packed: int) -> List[int]:
    return [(packed >> 4 * i) & 0xF for i in range(NUM_COLS)]


def _build_tables():
    """Precompute the result and score of sliding every possible row."""
    size = 1 << (4 * NUM_COLS)  # 65536 distinct rows
    left_result = [0] * size
    left_score = [0] * size
    right_result = [0] * size
    right_score = [0] * size
    can_merge = [False] * size

    for packed in range(size):
        row = _unpack_row(packed)

        # Sliding right is sliding left on the reversed row. The env writes it
        # as its own loop; verify_fast_env.py confirms the two agree.
        moved, score = _slide_row_left(row)
        left_result[packed] = _pack_row(moved)
        left_score[packed] = score

        moved, score = _slide_row_left(row[::-1])
        right_result[packed] = _pack_row(moved[::-1])
        right_score[packed] = score

        # _is_game_over compares raw adjacent cells, so two adjacent empties
        # count as mergeable there too. Match that.
        can_merge[packed] = any(row[i] == row[i + 1] for i in range(NUM_COLS - 1))

    return left_result, left_score, right_result, right_score, can_merge


(
    _ROW_LEFT,
    _ROW_LEFT_SCORE,
    _ROW_RIGHT,
    _ROW_RIGHT_SCORE,
    _ROW_CAN_MERGE,
) = _build_tables()

_ROW_MASK = 0xFFFF


def transpose(board: int) -> int:
    """Swap rows and columns of the packed board.

    Standard nibble-transpose trick: exchange the 2x2 blocks of nibbles, then
    the 2x2 blocks of those blocks.
    """
    a1 = board & 0xF0F00F0FF0F00F0F
    a2 = board & 0x0000F0F00000F0F0
    a3 = board & 0x0F0F00000F0F0000
    a = a1 | (a2 << 12) | (a3 >> 12)
    b1 = a & 0xFF00FF0000FF00FF
    b2 = a & 0x00FF00FF00000000
    b3 = a & 0x00000000FF00FF00
    return b1 | (b2 >> 24) | (b3 << 24)


def _slide_left(board: int) -> Tuple[int, int]:
    result = 0
    score = 0
    for row_index in range(NUM_ROWS):
        row = (board >> (16 * row_index)) & _ROW_MASK
        result |= _ROW_LEFT[row] << (16 * row_index)
        score += _ROW_LEFT_SCORE[row]
    return result, score


def _slide_right(board: int) -> Tuple[int, int]:
    result = 0
    score = 0
    for row_index in range(NUM_ROWS):
        row = (board >> (16 * row_index)) & _ROW_MASK
        result |= _ROW_RIGHT[row] << (16 * row_index)
        score += _ROW_RIGHT_SCORE[row]
    return result, score


def slide(board: int, action: int) -> Tuple[int, int]:
    """Apply one move. Returns (afterstate, merge score), no tile spawned yet.

    The afterstate (board after the slide, before the random tile appears) is
    what the n-tuple network learns to value.
    """
    if action == LEFT:
        return _slide_left(board)
    if action == RIGHT:
        return _slide_right(board)
    # Up/down are left/right on the transposed board.
    if action == UP:
        moved, score = _slide_left(transpose(board))
        return transpose(moved), score
    if action == DOWN:
        moved, score = _slide_right(transpose(board))
        return transpose(moved), score
    raise ValueError(f"unknown action {action}")


def empty_cells(board: int) -> List[int]:
    return [i for i in range(NUM_CELLS) if (board >> (4 * i)) & 0xF == 0]


def spawn_tile(board: int, rng: random.Random = random) -> int:
    """Place a 2 or 4 (equal probability) on a uniformly chosen empty cell.

    SGAI_2048._add_new rejection-samples a random (row, col) until it lands on
    an empty cell, which comes out uniform over the empty cells.
    """
    cells = empty_cells(board)
    if not cells:
        return board
    cell = rng.choice(cells)
    exponent = rng.choice((1, 2))
    return board | (exponent << (4 * cell))


def is_game_over(board: int) -> bool:
    """Match SGAI_2048._is_game_over: over only if full and nothing adjacent is equal."""
    transposed = transpose(board)
    for row_index in range(NUM_ROWS):
        row = (board >> (16 * row_index)) & _ROW_MASK
        column = (transposed >> (16 * row_index)) & _ROW_MASK
        if _ROW_CAN_MERGE[row] or _ROW_CAN_MERGE[column]:
            return False
    return not any((board >> (4 * i)) & 0xF == 0 for i in range(NUM_CELLS))


def legal_actions(board: int) -> List[int]:
    """Actions that actually change the board.

    Every action is accepted by the env (a no-op still costs NO_MOVE_PENALTY
    and still spawns a tile), but there is never a reason to pick one. An empty
    list means the game is over.
    """
    return [a for a in ACTIONS if slide(board, a)[0] != board]


def step(board: int, action: int, rng: random.Random = random) -> Tuple[int, int, bool]:
    """One full environment turn. Returns (next board, reward, terminated)."""
    afterstate, merge_score = slide(board, action)
    changed = afterstate != board
    next_board = spawn_tile(afterstate, rng)
    reward = merge_score if changed else NO_MOVE_PENALTY
    return next_board, reward, is_game_over(next_board)


def new_game(rng: random.Random = random) -> int:
    """A fresh board with two random starting tiles, as SGAI_2048.reset does."""
    return spawn_tile(spawn_tile(0, rng), rng)


def max_tile(board: int) -> int:
    """The largest tile value (not exponent) on the board."""
    best = max((board >> (4 * i)) & 0xF for i in range(NUM_CELLS))
    return 1 << best if best else 0


def pack(board_2d) -> int:
    """Convert a raw SGAI_2048 observation (tile values) to the packed form."""
    packed = 0
    for row_index in range(NUM_ROWS):
        for col_index in range(NUM_COLS):
            value = int(board_2d[row_index][col_index])
            exponent = value.bit_length() - 1 if value else 0
            packed |= exponent << (4 * (NUM_COLS * row_index + col_index))
    return packed


def unpack(board: int) -> List[List[int]]:
    """Convert the packed form back to a 2D list of tile values."""
    grid = []
    for row_index in range(NUM_ROWS):
        row = []
        for col_index in range(NUM_COLS):
            exponent = (board >> (4 * (NUM_COLS * row_index + col_index))) & 0xF
            row.append((1 << exponent) if exponent else 0)
        grid.append(row)
    return grid


def render(board: int) -> str:
    """A quick text dump of a packed board, handy when debugging."""
    return "\n".join(
        " ".join(f"{value:6d}" for value in row) for row in unpack(board)
    )
