import numpy as np
from environment import PlayerId, GameResult


SYMMETRIC_REWARDS_DICT = {'win': 1.0,
                          'loss': -1.0,
                          'draw': 0.0}

POSITIVE_REWARDS_DICT = {'win': 1.0,
                         'loss': 0.0,
                         'draw': 0.5}


def results_to_reward(game_result: GameResult, player_id: PlayerId, rewards_dict=None, reward_scale=1.0):
    if rewards_dict is None:
        rewards_dict = SYMMETRIC_REWARDS_DICT
    if game_result == GameResult.DRAW:
        reward = rewards_dict['draw']
    elif game_result == GameResult.PLAYER_X_WINS and player_id is PlayerId.X:
        reward = rewards_dict['win']
    elif game_result == GameResult.PLAYER_O_WINS and player_id is PlayerId.O:
        reward = rewards_dict['win']
    else:
        reward = rewards_dict['loss']
    return reward * reward_scale


def np_softmax(x):
    x_normed = x - np.max(x)
    y = np.exp(x_normed)
    f_x = y / np.sum(np.exp(x_normed))
    return f_x
