import numpy as np

from players.base_player import BasePlayer
from environment import TicTacToe


class RandomPlayer(BasePlayer):

    def next_move(self, board: TicTacToe):
        empty_slots = board.get_empty_slots()
        return empty_slots[np.random.randint(len(empty_slots))]

