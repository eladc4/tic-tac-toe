import numpy as np
from enum import Enum


class PlayerId(Enum):
    X = 'x'
    O = 'o'


class GameResult(Enum):
    DRAW = 0
    PLAYER_X_WINS = 1
    PLAYER_O_WINS = 2
    NOT_FINISHED = 4


class TicTacToe:

    def __init__(self):
        self.empty_sign = ' '
        self.board = np.array([self.empty_sign for _ in range(9)]).reshape((3, 3))

        self.win_lines = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                                   [0, 3, 6], [1, 4, 7], [2, 5, 8],
                                   [0, 4, 8], [2, 4, 6]])

    def check_win(self):
        flat_board = self.board.flatten()
        if np.any(np.sum(flat_board[self.win_lines] == PlayerId.X.value, axis=1) == 3):
            return GameResult.PLAYER_X_WINS
        if np.any(np.sum(flat_board[self.win_lines] == PlayerId.O.value, axis=1) == 3):
            return GameResult.PLAYER_O_WINS
        if all(flat_board != self.empty_sign):
            return GameResult.DRAW
        return GameResult.NOT_FINISHED

    def player_move(self, player_id: PlayerId, loc):
        if self.board[loc[0], loc[1]] != self.empty_sign:
            raise Exception('Illegal move')
        self.board[loc[0], loc[1]] = player_id.value

    def get_empty_slots(self):
        flat_board = self.board.flatten()
        return [np.unravel_index(_i, self.board.shape) for _i in range(self.board.size)
                if flat_board[_i] == self.empty_sign]

    def flatten(self):
        return ''.join(self.board.flatten())

    def print_board(self):
        print(f' {self.board[0,0]} | {self.board[0,1]} | {self.board[0,2]}')
        print('---+---+---')
        print(f' {self.board[1,0]} | {self.board[1,1]} | {self.board[1,2]}')
        print('---+---+---')
        print(f' {self.board[2,0]} | {self.board[2,1]} | {self.board[2,2]}')
