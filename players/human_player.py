import numpy as np

from players.base_player import BasePlayer
from environment import TicTacToe


class HumanPlayer(BasePlayer):

    def next_move(self, board: TicTacToe):
        move_str = input(f"player {self.id} type your move: ")
        return int(move_str[0])-1, int(move_str[-1])-1
