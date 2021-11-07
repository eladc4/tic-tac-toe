
import numpy as np
from typing import Dict, List
from collections import namedtuple

from players.base_player import BasePlayer
from players.common import results_to_reward, POSITIVE_REWARDS_DICT
from environment import TicTacToe, PlayerId


class TQPlayer(BasePlayer):
    """
    Player using the tabular Q-learning algorithm. Table implemented as a dictionary which keys as the
    flattened board as str, values are a list of length 9 that represent each move score.
    """

    def __init__(self, player_id, training=True, init_value=0.6, alpha=0.9, gamma=0.95):
        """

        :param player_id: either X or O
        :param training: train player while playing
        :param init_value: init value for Q-table entry. the initial score of a move
        :param alpha: learning rate
        :param gamma: reward discount factor
        """
        super().__init__(player_id)
        self.training = training
        self.init_value = init_value
        self.alpha = alpha
        self.gamma = gamma
        self.moves_dict = {}
        self.moves_list = []
        self.move = namedtuple('move_tuple', 'flat_board best_move_index')

    def next_move(self, board: TicTacToe):
        # flat board to use as key for moves_dict
        flat_board = board.flatten()

        # get moves values according to current board state. create a key if it doesn't exist
        if flat_board in self.moves_dict:
            move_values = self.moves_dict[flat_board]
        else:
            # init entry with init values, except illegal moves that get value -1
            move_values = np.ones(9) * self.init_value
            move_values[[c != board.empty_sign for c in flat_board]] = -1
            self.moves_dict[flat_board] = move_values

        # get best move
        best_move_index = np.argmax(move_values)
        best_move = np.unravel_index(best_move_index, board.board.shape)

        # insert move to moves_list memory
        self.moves_list.insert(0, self.move(flat_board, best_move_index))

        return best_move

    def end_game(self, game_result):
        """
        update values in Q-table according to game result. each move values are updated according to
        Q(S,A) = (1−α)∗Q(S,A) + α∗γ∗max_aQ(S′,a)
        S = current board state
        A = action. the selected move
        S′ = board state after selected move

        :param game_result:
        :return: None
        """
        if self.training:
            reward = results_to_reward(game_result, self.id, rewards_dict=POSITIVE_REWARDS_DICT)
            next_move_flat_board = None
            for i, move in enumerate(self.moves_list):
                if i == 0:
                    new_value = reward
                else:
                    new_value = self.alpha * self.moves_dict[move.flat_board][move.best_move_index] + \
                                (1-self.alpha) * self.gamma**i * np.max(self.moves_dict[next_move_flat_board])
                self.moves_dict[move.flat_board][move.best_move_index] = new_value
                next_move_flat_board = move.flat_board

        # reset moves list
        self.moves_list = []



