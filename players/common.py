import numpy as np
from environment import PlayerId


def results_to_reward(game_result, player_id):
    if game_result == 'draw':
        reward = 0.5
    elif game_result == 'player x wins' and player_id is PlayerId.X:
        reward = 1.0
    elif game_result == 'player o wins' and player_id is PlayerId.O:
        reward = 1.0
    else:
        reward = 0.0
    return reward


def np_softmax(x):
    x_normed = x - np.max(x)
    y = np.exp(x_normed)
    f_x = y / np.sum(np.exp(x_normed))
    return f_x
