import numpy as np
from typing import List
from tqdm import tqdm
from environment import TicTacToe, PlayerId, GameResult
from players import RandomPlayer, MinMaxPlayer, HumanPlayer, TQPlayer, NNPlayer
import matplotlib.pyplot as plt


def play_game(players: List, print_board=True):
    # init game
    ttt = TicTacToe()
    if print_board:
        ttt.print_board()

    # start game
    player_index = 0
    while ttt.check_win() is GameResult.NOT_FINISHED:
        ttt.player_move(players[player_index].id, players[player_index].next_move(ttt))
        if print_board:
            ttt.print_board()
            print('\n')
        player_index = (player_index + 1) % len(players)
    game_win_state = ttt.check_win()
    if print_board:
        print(game_win_state)
    [player.end_game(game_win_state) for player in players]
    return game_win_state


def run_games(players_list, number_of_games=100, print_board=False):
    game_results = [play_game(players_list, print_board=print_board) for _ in tqdm(range(number_of_games))]
    p_x_win = 100 * np.mean([g == GameResult.PLAYER_X_WINS for g in game_results])
    p_o_win = 100 * np.mean([g == GameResult.PLAYER_O_WINS for g in game_results])
    draw = 100 * np.mean([g == GameResult.DRAW for g in game_results])
    if players_list[0].id is PlayerId.X:
        return p_x_win, p_o_win, draw
    else:
        return p_o_win, p_x_win, draw


def print_results_figure(player_list, num_battles=100, num_games_per_battle=100, fig_name=None):
    result_table = np.zeros((num_battles, 3))
    for i_battle in range(num_battles):
        print(f'Battle {i_battle} / {num_battles}')
        result_table[i_battle, :] = run_games(player_list, number_of_games=num_games_per_battle, print_board=False)
    x = np.arange(num_battles)+1
    fig = plt.figure(1)
    plt.plot(x, result_table)
    plt.legend((f'p1 ({type(player_list[0]).__name__}) wins',
                f'p2 ({type(player_list[1]).__name__}) wins', 'draw'))
    plt.xlabel('battles')
    plt.ylabel('result prob [%]')
    if fig_name is not None:
        plt.title(fig_name)
    plt.show()

    return fig


def print_results_table(player_pairs_list, num_games=100):
    results = []
    for player_pair in player_pairs_list:
        players_str = f'{type(player_pair[0]).__name__}_{player_pair[0].id.value} - {type(player_pair[1]).__name__}_{player_pair[1].id.value}'
        print('Running game:', players_str)
        p1_win_prob, p2_win_prob, draw_prob = run_games(player_pair, number_of_games=N)
        results.append((p1_win_prob, p2_win_prob, draw_prob, players_str))
    print('N =', num_games)
    print('Player            | P1 Win | P2 Win  |  Draw')
    print('|:---|:---:|:---:|:---:|')
    for p1_win_prob, p2_win_prob, draw_prob, players_str in results:
        print(f'{players_str} | {p1_win_prob: 3.1f}% | {p2_win_prob: 3.1f}%  | {draw_prob: 3.1f}%')


if __name__ == '__main__':
    N = 2

    nn_player_x = NNPlayer(PlayerId.X, deterministic=False, lr=1e-4, random_move_prob=0.05)
    # nn_player_x = TQPlayer(PlayerId.X)
    players_list = [nn_player_x, RandomPlayer(PlayerId.O)]
    players_list_rev = [RandomPlayer(PlayerId.O), nn_player_x]
    fig = print_results_figure(players_list, num_battles=100)
    fig = print_results_figure(players_list_rev, num_battles=100)

    # nn_player_x = NNPlayer(PlayerId.X, deterministic=True)
    minmax_player_o = MinMaxPlayer(PlayerId.O, random_play_prob=0.1)
    players_list = [RandomPlayer(PlayerId.X), minmax_player_o]
    fig = print_results_figure(players_list, num_battles=100)
    players_list = [nn_player_x, minmax_player_o]
    fig = print_results_figure(players_list, num_battles=100)

    print(' start:')
    players_list = [HumanPlayer(PlayerId.O), nn_player_x]
    run_games(players_list, number_of_games=5, print_board=True)

    # init players
    minmax_player_x = MinMaxPlayer(PlayerId.X)
    minmax_player_o = MinMaxPlayer(PlayerId.O)
    random_player_x = RandomPlayer(PlayerId.X)
    random_player_o = RandomPlayer(PlayerId.O)
    tq_player_x = TQPlayer(PlayerId.X)
    tq_player_o = TQPlayer(PlayerId.O)
    nn_player_x = NNPlayer(PlayerId.X, deterministic=False)
    nn_player_o = NNPlayer(PlayerId.O, deterministic=False)

    player_pairs = [[nn_player_x, nn_player_o],
                    [nn_player_x, random_player_o],
                    [random_player_x, nn_player_o],
                    [nn_player_x, nn_player_o]]

    print_results_table(player_pairs, num_games=1000)

    player_pairs = [[random_player_x, random_player_o],
                    [minmax_player_x, random_player_o],
                    [random_player_o, minmax_player_x],
                    [random_player_x, minmax_player_o],
                    [minmax_player_x, minmax_player_o],
                    [tq_player_x, random_player_o],
                    [tq_player_x, minmax_player_o]]
    print_results_table(player_pairs, num_games=1000)

    print('done')
