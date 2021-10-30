
import numpy as np
from typing import Dict, List
from collections import namedtuple

import torch
from torch import nn

from players.common import results_to_reward
from players.base_player import BasePlayer
from environment import TicTacToe, PlayerId
from players.common import np_softmax


class NNPlayer(BasePlayer):

    def __init__(self, player_id, deterministic=True, gamma=0.95, lr=1e-4):
        super().__init__(player_id)
        self.deterministic = deterministic
        self.gamma = gamma
        self.moves_list = []
        self.move = namedtuple('move_tuple', 'flat_input move_logits best_move_index best_move_logit')
        self.model = self.build_model()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    @staticmethod
    def build_model():
        return nn.Sequential(nn.Linear(9*3, 9*9*3),
                             nn.ReLU(),
                             nn.Linear(9*9*3, 9),
                             )

    def next_move(self, board: TicTacToe):
        flat_board = board.flatten()
        if self.id is PlayerId.X:
            flat_input = np.concatenate([[float(p == PlayerId.X) for p in flat_board],
                                         [float(p == PlayerId.O) for p in flat_board],
                                         [float(p == board.empty_sign) for p in flat_board]], 0)
        else:
            flat_input = np.concatenate([[float(p == PlayerId.O) for p in flat_board],
                                         [float(p == PlayerId.X) for p in flat_board],
                                         [float(p == board.empty_sign) for p in flat_board]], 0)

        self.model.eval()
        move_logits = self.model(torch.Tensor(flat_input)).detach().numpy()
        _move_logits = move_logits.copy()
        _move_logits[[c != board.empty_sign for c in flat_board]] = -np.inf
        if self.deterministic:
            best_move_index = np.argmax(_move_logits)
        else:
            valid_move = False
            while not valid_move:
                best_move_index = np.random.choice(np.arange(len(_move_logits)), p=np_softmax(_move_logits))
                valid_move = flat_board[best_move_index] == board.empty_sign

        best_move = np.unravel_index(best_move_index, board.board.shape)

        self.moves_list.insert(0, self.move(flat_input, move_logits, best_move_index, move_logits[best_move_index]))
        return best_move

    def end_game(self, game_result):
        reward = results_to_reward(game_result, self.id)
        inputs, targets = [], []
        for i, move in enumerate(self.moves_list):
            inputs.append(move.flat_input)
            _target = move.move_logits.copy()
            if i == 0:
                _target[move.best_move_index] = reward
            else:
                _target[move.best_move_index] = self.gamma**i * self.moves_list[i-1].best_move_logit
            targets.append(_target)
        self.moves_list = []

        self.model.train()
        pred = self.model(torch.Tensor(np.stack(inputs)))
        loss = self.loss_fn(pred, torch.Tensor(np.stack(targets)))

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()






