"""TD(0) learning on afterstates for the n-tuple network.

The Q-learner in this repo learns Q(state, action), and every target it sees is
polluted by the tile that spawns after the move, so the same (state, action)
pair returns very different values depending on where the 2 or 4 landed.

A turn has two halves:

    board --(slide, deterministic, scores r)--> afterstate --(spawn)--> next board

Only the first half is under the agent's control, so this trainer learns
V(afterstate), the score still expected from here. Move selection is then

    argmax over legal a of [ r(board, a) + V(afterstate(board, a)) ]

with the afterstate TD(0) update from Szubert & Jaskowski (2014), "Temporal
difference learning of n-tuple networks for the game 2048":

    V(s'_t) <- V(s'_t) + alpha * [ r_{t+1} + V(s'_{t+1}) - V(s'_t) ]

where s'_t is the afterstate at turn t and r_{t+1}, s'_{t+1} come from the greedy
move at the next board. A terminal next board gives a target of 0.

No epsilon-greedy exploration. The spawn randomness already scatters the agent
across the state space, and forced random moves just slow things down here.

Temporal coherence (Jaskowski 2018) gives each weight its own learning rate,
since some entries are hit millions of times and others twice. Track the sum of
a weight's updates E and the sum of their absolute values A, then scale by
|E| / A: updates that all pull the same way keep a rate near 1, ones that cancel
out get throttled. Costs two extra arrays and several thousand points of score.
"""

import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from models.NTupleNetwork import NTupleNetwork
from utils import bitboard as bb


@dataclass
class GameResult:
    score: int
    max_tile: int
    moves: int


@dataclass
class TrainingStats:
    """Running tallies over a block of training games."""

    games: int = 0
    total_score: int = 0
    total_moves: int = 0
    best_score: int = 0
    best_tile: int = 0
    tile_counts: Dict[int, int] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def record(self, result: GameResult) -> None:
        self.games += 1
        self.total_score += result.score
        self.total_moves += result.moves
        self.best_score = max(self.best_score, result.score)
        self.best_tile = max(self.best_tile, result.max_tile)
        # Count a game as reaching every tile up to its largest, so
        # reach_rate(2048) means "fraction of games that got to 2048"
        tile = 2
        while tile <= result.max_tile:
            self.tile_counts[tile] = self.tile_counts.get(tile, 0) + 1
            tile *= 2

    @property
    def mean_score(self) -> float:
        return self.total_score / self.games if self.games else 0.0

    def reach_rate(self, tile: int) -> float:
        return self.tile_counts.get(tile, 0) / self.games if self.games else 0.0

    def moves_per_second(self) -> float:
        elapsed = time.time() - self.started_at
        return self.total_moves / elapsed if elapsed > 0 else 0.0


class NTupleTrainer:
    """Plays games against the fast simulator and learns from them."""

    def __init__(
        self,
        network: Optional[NTupleNetwork] = None,
        learning_rate: float = 1.0,
        temporal_coherence: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            network: the value function to train (a fresh one if omitted)
            learning_rate: alpha. 1.0 with temporal coherence on, since TC does
                the per-weight tuning; try around 0.1 with it off.
            temporal_coherence: enable per-weight adaptive learning rates
            seed: seed for the game RNG, for reproducible runs
        """
        self.network = network if network is not None else NTupleNetwork()
        self.learning_rate = learning_rate
        self.random = random.Random(seed)
        self.temporal_coherence = False
        self.update_sum = None
        self.update_abs_sum = None
        if temporal_coherence:
            self.enable_temporal_coherence()

    def enable_temporal_coherence(self, learning_rate: Optional[float] = None) -> None:
        """Switch TC on, allocating its two arrays the first time.

        Guei's OTD+TC runs in two phases: a long optimistic phase at a fixed
        rate, which keeps unvisited states looking attractive and drives
        exploration, then a short TC phase at the end that stops the weights
        oscillating and settles the policy. This is how the second phase starts.

        The counters begin empty, so every weight gets a full rate on its first
        TC update and is throttled from there as its own errors start to cancel.
        """
        if self.update_sum is None:
            # Same shape as the weights: signed and absolute update sums.
            self.update_sum = np.zeros_like(self.network.weights)
            self.update_abs_sum = np.zeros_like(self.network.weights)
        self.temporal_coherence = True
        if learning_rate is not None:
            self.learning_rate = learning_rate

    def save_optimizer(self, path: str) -> None:
        """Write the TC counters next to the weights.

        Resuming from weights alone restarts these at zero, which hands every
        weight a full learning rate again and undoes the fine-tuning TC spent
        the run building up. The first block after a resume visibly dips because
        of it. Saving them makes a resume continue rather than restart.
        """
        if self.update_sum is None:
            return
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez_compressed(
            path, update_sum=self.update_sum, update_abs_sum=self.update_abs_sum
        )

    def load_optimizer(self, path: str) -> None:
        with np.load(path) as data:
            update_sum = data["update_sum"]
            update_abs_sum = data["update_abs_sum"]
        if update_sum.shape != self.network.weights.shape:
            raise ValueError("optimizer state does not match the network's weights")
        self.update_sum = update_sum.astype(np.float32, copy=False)
        self.update_abs_sum = update_abs_sum.astype(np.float32, copy=False)
        self.temporal_coherence = True

    # the learning rule

    def _scatter_add(self, array: np.ndarray, indices, amounts) -> None:
        """array[indices] += amounts, correct even when indices repeat.

        Two of a pattern's 8 symmetric placements can land on the same weight
        when the board itself is symmetric, and plain fancy indexing would apply
        only one of the updates. np.add.at handles that but is several times
        slower, so check for duplicates first; almost every step has none.
        """
        if len(set(indices)) == len(indices):
            array[indices] += amounts
        else:
            np.add.at(array, indices, amounts)

    def _update(self, indices: List[int], td_error: float) -> None:
        """Nudge every weight this afterstate touches toward the TD target."""
        per_weight = td_error / self.network.num_lookups

        if not self.temporal_coherence:
            self._scatter_add(
                self.network.weights, indices, np.float32(self.learning_rate * per_weight)
            )
            return

        absolute_sums = self.update_abs_sum[indices]
        # A weight with no history yet learns at the full rate
        rates = np.where(
            absolute_sums > 0,
            np.abs(self.update_sum[indices]) / np.maximum(absolute_sums, 1e-12),
            1.0,
        )
        self._scatter_add(
            self.network.weights,
            indices,
            (self.learning_rate * rates * per_weight).astype(np.float32),
        )
        self._scatter_add(self.update_sum, indices, np.float32(per_weight))
        self._scatter_add(self.update_abs_sum, indices, np.float32(abs(per_weight)))

    # playing and learning

    def _best_move(self, board: int):
        """Greedy move: the legal action maximizing reward + V(afterstate).

        Returns (afterstate, reward, value, indices), or None when the board is
        terminal. Only board-changing moves count. A no-op is always available
        in this env but costs 10 points and spawns a tile anyway.
        """
        best = None
        best_total = -float("inf")
        network = self.network
        for action in bb.ACTIONS:
            afterstate, reward = bb.slide(board, action)
            if afterstate == board:
                continue
            indices = network.indices(afterstate)
            value = network.value_from_indices(indices)
            total = reward + value
            if total > best_total:
                best_total = total
                best = (afterstate, reward, value, indices)
        return best

    def play_game(self, learn: bool = True) -> GameResult:
        """Play one full game, updating the network as it goes."""
        board = bb.new_game(self.random)
        score = 0
        moves = 0
        previous_indices: Optional[List[int]] = None

        while True:
            choice = self._best_move(board)

            if choice is None:
                # Terminal: no further score arrives, so the target is 0.
                if learn and previous_indices is not None:
                    previous_value = self.network.value_from_indices(previous_indices)
                    self._update(previous_indices, -previous_value)
                break

            afterstate, reward, value, indices = choice

            if learn and previous_indices is not None:
                # Recompute the previous afterstate's value. Consecutive
                # afterstates share patterns, so last step's cached value may
                # already be stale.
                previous_value = self.network.value_from_indices(previous_indices)
                self._update(previous_indices, reward + value - previous_value)

            previous_indices = indices
            score += reward
            moves += 1
            board = bb.spawn_tile(afterstate, self.random)

        return GameResult(score=score, max_tile=bb.max_tile(board), moves=moves)

    def train(self, num_games: int, stats: Optional[TrainingStats] = None) -> TrainingStats:
        stats = stats if stats is not None else TrainingStats()
        for _ in range(num_games):
            stats.record(self.play_game(learn=True))
        return stats

    def evaluate(self, num_games: int) -> TrainingStats:
        """Play without learning, to measure the policy as it currently stands."""
        stats = TrainingStats()
        for _ in range(num_games):
            stats.record(self.play_game(learn=False))
        return stats
