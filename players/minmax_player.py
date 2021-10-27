import numpy as np
from typing import Dict, List

from players.base_player import BasePlayer
from environment import TicTacToe, PlayerId


class MinMaxPlayer(BasePlayer):

    def next_move(self, board: TicTacToe):
        moves_tree = self._build_moves_tree(board, self.id)
        return self._get_best_move(moves_tree)[0]

    def _build_moves_tree(self, board: TicTacToe, player_id: PlayerId):
        moves_list = []
        next_player_id = PlayerId.O if player_id is PlayerId.X else PlayerId.X
        for empty_slot in board.get_empty_slots():
            board.player_move(player_id, empty_slot)
            win_state = board.check_win()
            if win_state == 'no win':
                result = self._build_moves_tree(board, next_player_id)
            elif win_state == 'player x wins':
                result = 1 if self.id is PlayerId.X else -1
            elif win_state == 'player o wins':
                result = 1 if self.id is PlayerId.O else -1
            else:  # win_state == 'draw'
                result = 0
            moves_list.append((player_id, empty_slot, board.flatten(), result))
            board.board[empty_slot[0], empty_slot[1]] = board.empty_sign
        return moves_list

    def _get_best_move(self, moves_tree: List):
        results = []
        for leaf in moves_tree:
            res = leaf[3]
            result = res if isinstance(res, int) else self._get_best_move(leaf[3])[1]
            results.append(result)
        best_move_score = max(results) if moves_tree[0][0] is self.id else min(results)
        best_move_index = np.argmax(results) if moves_tree[0][0] is self.id else np.argmin(results)
        best_move = moves_tree[best_move_index][1]
        return best_move, best_move_score




