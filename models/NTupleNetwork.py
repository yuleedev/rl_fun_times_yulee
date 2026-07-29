"""An n-tuple network: the value function that replaces the Q-table.

A tabular Q-learner stores one entry per whole board, and there are far more
reachable 4x4 boards than any training run will visit. That is why
QAgentInference falls back to random moves so often: almost every board it sees
at play time is new to it.

An n-tuple network stores one entry per small patch instead. Each "tuple" here
is 6 cells, so a patch has at most 16^6 configurations, which a training run can
actually cover. A board's value is the sum of the weights of all its patches, so
an unseen board still gets a sensible value from patches seen thousands of times.

Two things on top of that:

* Symmetry. A board and its 8 rotations/mirrors are the same position (see
  utils/state_keys.py). Instead of canonicalizing the board, each pattern is
  applied in all 8 orientations sharing one weight table, so one observation
  trains all 8.
* Afterstates. The network values the board after the slide but before the tile
  spawns, which removes the spawn's randomness from the target. See
  models/NTupleTrainer.py.

The default 4 patterns are the standard set from Szubert & Jaskowski: two
straight 6-cell strips and two 2x3 blocks. 4 patterns x 8 symmetries = 32 weight
reads per board evaluation.
"""

import os
from array import array
from typing import List, Sequence, Tuple

import numpy as np

from utils.bitboard import NUM_CELLS, NUM_COLS, NUM_ROWS

# Cell numbering is row-major, matching utils.bitboard:
#    0  1  2  3
#    4  5  6  7
#    8  9 10 11
#   12 13 14 15
DEFAULT_PATTERNS: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5),  # top row plus the two cells below its left end
    (4, 5, 6, 7, 8, 9),  # second row plus two of the third
    (0, 1, 2, 4, 5, 6),  # 2x3 block in the top-left corner
    (4, 5, 6, 8, 9, 10),  # 2x3 block one row down
)

ALPHABET = 16  # a cell holds an exponent 0..15, so 16 values per tuple slot


def _symmetry_maps() -> List[np.ndarray]:
    """The 8 symmetries of a square, as permutations of the 16 cell indices."""
    base = np.arange(NUM_CELLS).reshape(NUM_ROWS, NUM_COLS)
    maps = []
    for num_rotations in range(4):
        rotated = np.rot90(base, num_rotations)
        maps.append(rotated.ravel())
        maps.append(np.fliplr(rotated).ravel())
    return maps


def _compile_extractor(shift_groups: Sequence[Sequence[int]], offsets: Sequence[int]):
    """Generate a flat function that reads all pattern indices out of a board.

    This is the hottest code in training, so rather than looping over patterns
    at runtime the shift/mask arithmetic is written out once as source and
    compiled. The generated body is just:

        def extract(board):
            return [
                0 + (((board >> 0) & 15) | ((board >> 4) & 15) << 4 | ...),
                ...
            ]

    Each entry indexes the single flat weight array; the per-pattern offset is
    folded in so callers never have to know which table they hit.
    """
    lines = ["def extract(board):", "    return ["]
    for shifts, offset in zip(shift_groups, offsets):
        terms = " | ".join(
            f"(((board >> {shift}) & 15) << {4 * slot})"
            for slot, shift in enumerate(shifts)
        )
        lines.append(f"        {offset} + ({terms}),")
    lines.append("    ]")
    namespace: dict = {}
    exec("\n".join(lines), namespace)  # noqa: S102 - generated from constants above
    return namespace["extract"]


def _compile_value(shift_groups: Sequence[Sequence[int]], offsets: Sequence[int]):
    """Generate a function that reads a board's value in one expression.

    `value` above builds a 32-element list and hands it to numpy, which costs
    an allocation and a fancy-index dispatch. Search calls this hundreds of
    thousands of times per game and cares about nothing else, so the fast path
    inlines the index arithmetic straight into a sum of scalar lookups:

        def value(board, w):
            return w[0 + (...)] + w[16777216 + (...)] + ...

    Indexing a plain `array('f')` returns a Python float far quicker than numpy
    returns a scalar, so the whole evaluation stays in the interpreter with no
    temporaries. Training keeps using the numpy weights, which is why this is a
    separate opt-in path rather than the default.
    """
    terms = []
    for shifts, offset in zip(shift_groups, offsets):
        index = " | ".join(
            f"(((board >> {shift}) & 15) << {4 * slot})"
            for slot, shift in enumerate(shifts)
        )
        terms.append(f"w[{offset} + ({index})]")
    source = "def value(board, w):\n    return " + " + ".join(terms) + "\n"
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102 - generated from constants above
    return namespace["value"]


class NTupleNetwork:
    """A sum-of-lookups value function over packed bitboards."""

    def __init__(self, patterns: Sequence[Sequence[int]] = DEFAULT_PATTERNS):
        self.patterns = tuple(tuple(pattern) for pattern in patterns)
        sizes = {len(pattern) for pattern in self.patterns}
        if len(sizes) != 1:
            raise ValueError("all patterns must have the same length")
        self.tuple_length = sizes.pop()
        self.table_size = ALPHABET**self.tuple_length

        # Each pattern gets one table; its 8 symmetric placements share it.
        shift_groups: List[List[int]] = []
        offsets: List[int] = []
        for pattern_index, pattern in enumerate(self.patterns):
            for symmetry in _symmetry_maps():
                shift_groups.append([4 * int(symmetry[cell]) for cell in pattern])
                offsets.append(pattern_index * self.table_size)
        self.num_lookups = len(shift_groups)  # 32 with the default patterns

        self._extract = _compile_extractor(shift_groups, offsets)
        self._compile_fast_value = lambda: _compile_value(shift_groups, offsets)
        self._fast_value = None
        self._fast_weights = None
        self.weights = np.zeros(len(self.patterns) * self.table_size, dtype=np.float32)

    # evaluation

    def indices(self, board: int) -> List[int]:
        """The weight-array positions this board touches (one per pattern placement)."""
        return self._extract(board)

    def enable_fast_value(self) -> None:
        """Switch `value` to the compiled scalar path, for search.

        Takes a snapshot of the weights into an array('f'), so call it after
        loading and never while training: updates go to the numpy array and
        will not be seen here.
        """
        weights = array("f")
        weights.frombytes(self.weights.astype(np.float32, copy=False).tobytes())
        self._fast_weights = weights
        self._fast_value = self._compile_fast_value()

    def value(self, board: int) -> float:
        """Estimated remaining score from this board."""
        if self._fast_value is not None:
            return self._fast_value(board, self._fast_weights)
        return float(self.weights[self._extract(board)].sum())

    def value_from_indices(self, indices: List[int]) -> float:
        return float(self.weights[indices].sum())

    # persistence

    def save(self, path: str) -> None:
        """Write weights plus the pattern set, so loading can't silently mismatch."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # Most of the table is still zero, which compresses to almost nothing.
        np.savez_compressed(
            path,
            weights=self.weights,
            patterns=np.array(self.patterns, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str) -> "NTupleNetwork":
        with np.load(path) as data:
            patterns = [tuple(int(cell) for cell in row) for row in data["patterns"]]
            network = cls(patterns=patterns)
            weights = data["weights"]
            if weights.shape != network.weights.shape:
                raise ValueError(
                    f"weight file has {weights.shape[0]} entries, patterns need "
                    f"{network.weights.shape[0]}"
                )
            network.weights = weights.astype(np.float32, copy=False)
        return network

    def optimistic_init(self, initial_value: float) -> None:
        """Start every weight high, so unseen states look attractive.

        Guei's optimistic initialization. 2048 was long assumed to need no
        explicit exploration, since the random spawns scatter the agent anyway,
        and epsilon-greedy actively hurts here because it breaks the greedy
        behaviour the value function is learned under. Optimistic init explores
        without ever acting randomly: an unvisited state still carries its full
        initial value, so the greedy policy walks toward it on its own, and the
        value decays to the truth only once the state has been played out.

        Every board reads as `initial_value` at the start, so each of the
        num_lookups weights gets an equal share of it.
        """
        self.weights[:] = np.float32(initial_value / self.num_lookups)

    def memory_bytes(self) -> int:
        return self.weights.nbytes

    def coverage(self) -> float:
        """Fraction of weights that training has actually touched."""
        return float(np.count_nonzero(self.weights)) / self.weights.size
