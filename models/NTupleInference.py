"""Inference agent for the n-tuple network + expectimax.

Same get_action(obs_raw=...) interface as QAgentInference and DQAgentInference,
so it drops into the existing runners:

    from models.NTupleInference import NTupleInference
    rl_agent = NTupleInference(
        weights_file_path=os.path.join("models", "ntuple_files", "ntuple_final.npz"),
        search_depth=3,
    )

Two knobs at play time:

    node_budget   search spent per move. 0 turns search off and plays the greedy
                  policy the trainer optimizes, which is instant and already
                  decent. Raise it to trade time for strength.
    max_depth     ceiling on how deep the budget can push.

Depth is picked per board from the budget (utils.expectimax.depth_for_board),
since a fixed depth costs very different amounts on a crowded board vs an open
one.
"""

import random
from typing import Optional

import numpy as np

from models.NTupleNetwork import NTupleNetwork
from utils import bitboard as bb
from utils.expectimax import Expectimax, depth_for_board


class NTupleInference:

    def __init__(
        self,
        weights_file_path: str,
        node_budget: int = 2_000,
        max_depth: int = 5,
        tile_downgrading: bool = True,
        seed: Optional[int] = None,
    ):
        self.network = NTupleNetwork.load(weights_file_path)
        # Inference never updates the weights, so the compiled scalar path is
        # safe here and shaves time off every node the search visits.
        self.network.enable_fast_value()
        self.node_budget = node_budget
        self.max_depth = max_depth
        self.random = random.Random(seed)
        self.searcher = Expectimax(
            self.network, depth=2, tile_downgrading=tile_downgrading
        )

    def get_action(self, obs_raw: np.ndarray) -> int:
        board = bb.pack(obs_raw)

        if self.node_budget <= 0 or self.max_depth <= 1:
            return self._greedy_action(board)

        self.searcher.depth = depth_for_board(board, self.max_depth, self.node_budget)
        return self.searcher.best_action(board, self.random)

    def _greedy_action(self, board: int) -> int:
        """One ply: maximize immediate reward plus the afterstate's value."""
        best_action = None
        best_value = -float("inf")
        for action in bb.ACTIONS:
            afterstate, reward = bb.slide(board, action)
            if afterstate == board:
                continue  # no-op: -10 points and a spawned tile for nothing
            value = reward + self.network.value(afterstate)
            if value > best_value:
                best_value = value
                best_action = action
        if best_action is None:
            return self.random.choice(bb.ACTIONS)  # dead board, env ends the game
        return best_action
