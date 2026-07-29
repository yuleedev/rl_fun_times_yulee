"""Expectimax search on top of the learned value function.

2048 has no opponent, only dice, so the tree alternates between MAX nodes (our
turn, best of the board-changing moves) and CHANCE nodes (the spawn, averaged
over where a tile could appear).

The spawn model matches this env rather than standard 2048: the new tile is a 2
or a 4 at 50/50 (_add_new does random.choice([2, 4]), not the usual 90/10), and
the cell is uniform over the empty ones. Getting that wrong makes the search
walk into positions that only hold up if 4s are rare.

Leaves are evaluated with the n-tuple network.

A chance node with k empty cells has 2k children, each branching into up to 4
moves, so one extra ply multiplies the tree by about 8k. A fixed depth is
therefore a bad knob: depth 3 is instant on a nearly-full board and takes
minutes per move on an open one. depth_for_board picks the depth from a node
budget instead, keeping the time per move roughly constant.
"""

import random
from typing import Dict, Tuple

from models.NTupleNetwork import NTupleNetwork
from utils import bitboard as bb

# Branches rarer than this add almost nothing to the average but cost as much as
# any other subtree, so evaluate them with the network instead of expanding.
DEFAULT_PROBABILITY_FLOOR = 0.0005

# Only remap boards whose largest tile is at least this rare. Training reaches
# 4096 in most games, so those weights are fine; 8192 shows up in a few percent.
DEFAULT_DOWNGRADE_EXPONENT = 13  # 8192


def downgrade(board: int, min_exponent: int = DEFAULT_DOWNGRADE_EXPONENT) -> int:
    """Remap a board with rare high tiles onto a better trained equivalent.

    From Guei's tile-downgrading. The network barely trained on boards holding
    an 8192, so its opinion of them is close to noise, which is exactly where
    the game is won or lost. If some tile value is *missing* below the maximum,
    every tile above that gap can be halved: the gap guarantees the halved
    values still land above everything left alone, so no two tiles become equal
    that were not equal before. The board is structurally identical, the legal
    moves and merges correspond one to one, and it sits in a region the network
    actually knows.

    Search the downgraded board, then play the chosen action on the real one.
    Rewards inside the search come out halved too, but the whole search is on
    one scale, and only the argmax matters.
    """
    for _ in range(bb.MAX_EXPONENT):
        exponents = {
            (board >> (4 * cell)) & 0xF for cell in range(bb.NUM_CELLS)
        } - {0}
        if not exponents:
            return board
        top = max(exponents)
        if top < min_exponent:
            return board

        gap = next((e for e in range(top - 1, 0, -1) if e not in exponents), None)
        if gap is None:
            return board

        remapped = 0
        for cell in range(bb.NUM_CELLS):
            exponent = (board >> (4 * cell)) & 0xF
            if exponent > gap:
                exponent -= 1
            remapped |= exponent << (4 * cell)
        board = remapped
    return board


class Expectimax:
    """Depth-limited expectimax with a transposition table."""

    def __init__(
        self,
        network: NTupleNetwork,
        depth: int = 3,
        probability_floor: float = DEFAULT_PROBABILITY_FLOOR,
        tile_downgrading: bool = True,
        downgrade_exponent: int = DEFAULT_DOWNGRADE_EXPONENT,
    ):
        """
        Args:
            network: value function used at the leaves
            depth: how many of our own moves to look ahead. 1 means no search,
                i.e. greedy on reward + V, the policy the trainer learns under.
            probability_floor: stop expanding a branch once the chance of
                reaching it drops below this
            tile_downgrading: remap rare high-tile boards before searching
            downgrade_exponent: smallest top tile worth remapping, as an exponent
        """
        self.network = network
        self.depth = depth
        self.probability_floor = probability_floor
        self.tile_downgrading = tile_downgrading
        self.downgrade_exponent = downgrade_exponent
        self._max_cache: Dict[Tuple[int, int], float] = {}
        self._chance_cache: Dict[Tuple[int, int], float] = {}
        self._leaf_cache: Dict[int, float] = {}

    def best_action(self, board: int, rng: random.Random = random) -> int:
        """The action to play from this board."""
        # Positions repeat constantly within one search and almost never across
        # searches, so the tables only need to live as long as this call.
        self._max_cache.clear()
        self._chance_cache.clear()
        self._leaf_cache.clear()

        # Downgrading preserves which tiles are equal, so the move that is best
        # on the remapped board is the move to play on the real one.
        if self.tile_downgrading:
            board = downgrade(board, self.downgrade_exponent)

        best_action = None
        best_value = -float("inf")
        for action in bb.ACTIONS:
            afterstate, reward = bb.slide(board, action)
            if afterstate == board:
                continue  # no-op move: costs 10 points and spawns a tile anyway
            value = reward + self._chance_value(afterstate, self.depth - 1, 1.0)
            if value > best_value:
                best_value = value
                best_action = action

        if best_action is None:
            # Dead board; the env reports terminated either way.
            return rng.choice(bb.ACTIONS)
        return best_action

    def _chance_value(self, afterstate: int, depth: int, probability: float) -> float:
        """Expected value of an afterstate, averaging over the tile about to spawn."""
        if depth <= 0 or probability < self.probability_floor:
            # Leaves repeat heavily across the tree, and a network eval costs far
            # more than a dict lookup, so cache them like the interior nodes.
            cached = self._leaf_cache.get(afterstate)
            if cached is None:
                cached = self.network.value(afterstate)
                self._leaf_cache[afterstate] = cached
            return cached

        key = (afterstate, depth)
        cached = self._chance_cache.get(key)
        if cached is not None:
            return cached

        cells = bb.empty_cells(afterstate)
        if not cells:
            # Full board: nothing spawns, so it is our turn again immediately.
            return self._max_value(afterstate, depth, probability)

        cell_probability = 0.5 / len(cells)  # uniform cell, then 2 or 4 at 50/50
        total = 0.0
        for cell in cells:
            shift = 4 * cell
            for exponent in (1, 2):
                child = afterstate | (exponent << shift)
                total += self._max_value(
                    child, depth, probability * cell_probability
                )
        total *= cell_probability

        self._chance_cache[key] = total
        return total

    def _max_value(self, board: int, depth: int, probability: float) -> float:
        """Best value reachable from a board it is our turn to move from."""
        key = (board, depth)
        cached = self._max_cache.get(key)
        if cached is not None:
            return cached

        best = None
        for action in bb.ACTIONS:
            afterstate, reward = bb.slide(board, action)
            if afterstate == board:
                continue
            value = reward + self._chance_value(afterstate, depth - 1, probability)
            if best is None or value > best:
                best = value

        result = 0.0 if best is None else best  # no legal move means game over
        self._max_cache[key] = result
        return result


def depth_for_board(board: int, max_depth: int, node_budget: int) -> int:
    """Deepest search this board can afford within roughly node_budget leaves.

    Each extra ply multiplies the tree by about 8k, where k is the number of
    empty cells (2k spawn outcomes, each branching into up to 4 moves), so the
    deepest affordable depth solves (8k)^(depth-1) <= node_budget.

    Crowded boards have few empty cells, which makes deep search both cheap and
    worth having, since that is where games are lost. Open boards get depth 2.

    Actual cost lands under this estimate, since the transposition table and the
    probability floor cut branches out. Treat node_budget as a dial, not a
    guarantee.
    """
    empties = len(bb.empty_cells(board))
    if empties == 0:
        return max_depth
    branching = 8 * empties
    depth = 2
    estimated_nodes = branching
    while depth < max_depth and estimated_nodes * branching <= node_budget:
        estimated_nodes *= branching
        depth += 1
    return depth
