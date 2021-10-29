
import numpy as np
from typing import Dict, List
from collections import namedtuple

from players.base_player import BasePlayer
from environment import TicTacToe, PlayerId


class TQPlayer(BasePlayer):

    def __init__(self, player_id, init_value=0.6, alpha=0.9, gamma=0.95):
        super().__init__(player_id)
        self.init_value = init_value
        self.alpha = alpha
        self.gamma = gamma
        self.moves_dict = {}#defaultdict(lambda: init_value*np.ones(9))
        self.moves_list = []
        self.move = namedtuple('move_tuple', 'flat_board best_move_index')

    def next_move(self, board: TicTacToe):
        flat_board = board.flatten()
        if flat_board in self.moves_dict:
            move_values = self.moves_dict[flat_board]
        else:
            move_values = np.ones(9) * self.init_value
            move_values[[c != board.empty_sign for c in flat_board]] = -1
            self.moves_dict[flat_board] = move_values
        best_move_index = np.argmax(move_values)
        best_move = np.unravel_index(best_move_index, board.board.shape)
        self.moves_list.insert(0, self.move(flat_board, best_move_index))
        return best_move

    def end_game(self, game_result):
        if game_result == 'draw':
            reward = 0.5
        elif game_result == 'player x wins' and self.id is PlayerId.X:
            reward = 1.0
        elif game_result == 'player o wins' and self.id is PlayerId.O:
            reward = 1.0
        else:
            reward = 0.0

        next_move_flat_board = None
        for i, move in enumerate(self.moves_list):
            if i == 0:
                new_value = reward
            else:
                new_value = self.alpha * self.moves_dict[move.flat_board][move.best_move_index] + \
                            (1-self.alpha) * self.gamma**i * np.max(self.moves_dict[next_move_flat_board])
            self.moves_dict[move.flat_board][move.best_move_index] = new_value
            next_move_flat_board = move.flat_board
        self.moves_list = []



