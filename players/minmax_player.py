import numpy as np
from typing import Dict, List

from players.base_player import BasePlayer
from environment import TicTacToe, PlayerId, GameResult


class MinMaxPlayer(BasePlayer):
    """
    Player using the min-max algorithm
    """
    def __init__(self, player_id, random_play_prob=0.0):
        """
        init min-max player (optimal player)
        :param player_id: either X or O
        :param random_play_prob: probability of playing a random move instead of the optimal move
        """
        super().__init__(player_id)
        self.random_play_prob = random_play_prob
        self.moves_dict = {}

    def next_move(self, board: TicTacToe):
        # get move from moves_dict, if not there yet. calculate it
        best_move = self.moves_dict.get(board.flatten())
        if best_move is None:
            # build all possible moves tree
            moves_tree = self._build_moves_tree(board, self.id)
            # collapse moves tree to calculate best move
            best_move = self._get_best_move(moves_tree)[0]
            # store best move in moves_dict
            self.moves_dict.update({board.flatten(): best_move})

        if np.random.rand() < self.random_play_prob:
            empty_slots = board.get_empty_slots()
            return empty_slots[np.random.randint(len(empty_slots))]
        else:
            return best_move

    def _build_moves_tree(self, board: TicTacToe, player_id: PlayerId):
        """
        build all possible moves tree from input board, recursively.
        tree is implemented as a list of lists. each element is either a leaf if the game is over or
        a list containing the next nodes
        a leaf is a tuple: (player_id, move, flat board as str, result)
        :param board: TicTacToe object
        :param player_id: current player id, either X or O
        :return: a tree of all possible moves
        """
        moves_list = []
        next_player_id = PlayerId.O if player_id is PlayerId.X else PlayerId.X
        # go over all possible moves in current board state
        for empty_slot in board.get_empty_slots():
            # play move
            board.player_move(player_id, empty_slot)

            win_state = board.check_win()
            if win_state == GameResult.NOT_FINISHED:
                result = self._build_moves_tree(board, next_player_id)
            elif win_state == GameResult.PLAYER_X_WINS:
                result = 1 if self.id is PlayerId.X else -1
            elif win_state == GameResult.PLAYER_O_WINS:
                result = 1 if self.id is PlayerId.O else -1
            else:  # win_state == GameResult.DRAW
                result = 0

            # add new node to tree
            moves_list.append((player_id, empty_slot, board.flatten(), result))
            # reset move
            board.board[empty_slot[0], empty_slot[1]] = board.empty_sign
        return moves_list

    def _get_best_move(self, moves_tree: List):
        """
        collapses a move_tree list to calculate the best move
        :param moves_tree: move tree
        :return: best move as tuple (x, y) and best move score
        """
        results = []
        for leaf in moves_tree:
            # go over all moves in current state
            res = leaf[3]
            # calc node value
            result = res if isinstance(res, int) else self._get_best_move(leaf[3])[1]
            # add to values list
            results.append(result)

        # apply min\max according to current player id for score and index
        best_move_score = max(results) if moves_tree[0][0] is self.id else min(results)
        best_move_index = np.argmax(results) if moves_tree[0][0] is self.id else np.argmin(results)
        best_move = moves_tree[best_move_index][1]
        return best_move, best_move_score



