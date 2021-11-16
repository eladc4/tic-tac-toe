import numpy as np
from enum import Enum

from environment import PlayerId, GameResult


class Result(Enum):
    WIN = 'win'
    LOSS = 'loss'
    DRAW = 'draw'


SYMMETRIC_REWARDS_DICT = {Result.WIN: 1.0,
                          Result.LOSS: -1.0,
                          Result.DRAW: 0.0}

POSITIVE_REWARDS_DICT = {Result.WIN: 1.0,
                         Result.LOSS: 0.0,
                         Result.DRAW: 0.5}


def results_to_reward(game_result: GameResult, player_id: PlayerId, rewards_dict=None, reward_scale=1.0):
    if rewards_dict is None:
        rewards_dict = SYMMETRIC_REWARDS_DICT
    if game_result == GameResult.DRAW:
        result = Result.DRAW
    elif game_result == GameResult.PLAYER_X_WINS and player_id is PlayerId.X:
        result = Result.WIN
    elif game_result == GameResult.PLAYER_O_WINS and player_id is PlayerId.O:
        result = Result.WIN
    else:
        result = Result.LOSS
    reward = rewards_dict[result]

    return reward * reward_scale


def np_softmax(x):
    x_normed = x - np.max(x)
    y = np.exp(x_normed)
    f_x = y / np.sum(np.exp(x_normed))
    return f_x
