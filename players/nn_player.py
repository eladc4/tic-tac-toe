
import numpy as np
from typing import Dict, List
from collections import namedtuple

import torch
from torch import nn

from players.common import results_to_reward
from players.base_player import BasePlayer
from environment import TicTacToe, PlayerId, GameResult
from players.common import np_softmax


class NNPlayer(BasePlayer):
    """
    Player using the Neural Network Q-learning algorithm.
    """

    def __init__(self, player_id, deterministic=True, training=True, gamma=0.95, lr=1e-4, reward_scale=100.0):
        """

        :param player_id: either X or O
        :param deterministic: if True (default) always play the move with best score. if False select
         a move according to scores distribution
        :param training: train player while playing
        :param gamma: reward discount factor
        :param lr: learning rate for training the player
        """
        super().__init__(player_id)
        self.deterministic = deterministic
        self.training = training
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.moves_list = []
        self.move = namedtuple('move_tuple', 'flat_input move_logits best_move_index best_move_logit')
        self.model = self.build_model()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    @staticmethod
    def build_model():
        """
        Builds the player network
        :return: Torch Model
        """
        return nn.Sequential(nn.Linear(9*3, 9*9*3),
                             nn.ReLU(),
                             # nn.Linear(9 * 9 * 3, 9 * 9 * 3),
                             # nn.ReLU(),
                             nn.Linear(9*9*3, 9),
                             )

    def next_move(self, board: TicTacToe):
        # build input to network: 3x3 x 3 channels for each player id => (player id, opponent id, empty slot)
        flat_board = board.flatten()
        if self.id is PlayerId.X:
            flat_input = np.concatenate([[float(p == PlayerId.X) for p in flat_board],
                                         [float(p == PlayerId.O) for p in flat_board],
                                         [float(p == board.empty_sign) for p in flat_board]], 0)
        else:
            flat_input = np.concatenate([[float(p == PlayerId.O) for p in flat_board],
                                         [float(p == PlayerId.X) for p in flat_board],
                                         [float(p == board.empty_sign) for p in flat_board]], 0)

        # run model inference for moves calculation
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

        # save move for training
        self.moves_list.insert(0, self.move(flat_input, move_logits.copy(), best_move_index, move_logits[best_move_index]))
        return best_move

    def end_game(self, game_result: GameResult):
        """
        train model according to saved moves and game result
        :param game_result: game result
        :return: None
        """
        if self.training:
            reward = results_to_reward(game_result, self.id, reward_scale=self.reward_scale)

            # build training batch: inputs and targets according to game result
            inputs, targets = [], []
            for i, move in enumerate(self.moves_list):
                inputs.append(move.flat_input)
                _target = move.move_logits.copy()
                if i == 0:
                    # last move
                    _target[move.best_move_index] = reward
                else:
                    _target[move.best_move_index] = self.gamma**i * self.moves_list[i-1].best_move_logit
                targets.append(_target)

            self.model.train()
            # infer game inputs
            pred = self.model(torch.Tensor(np.stack(inputs)))
            # calculate loss
            loss = self.loss_fn(pred, torch.Tensor(np.stack(targets)))

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()  # update network weights

        # reset move list
        self.moves_list = []







