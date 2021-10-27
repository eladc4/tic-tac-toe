import numpy as np
from typing import List
from environment import TicTacToe, PlayerId
from players import RandomPlayer, MinMaxPlayer, HumanPlayer


def play_game(players: List, print_board=True):
    # init game
    ttt = TicTacToe()

    # start game
    player_index = 0
    while ttt.check_win() == 'no win':
        ttt.player_move(players[player_index].id, players[player_index].next_move(ttt))
        if print_board:
            ttt.print_board()
        player_index = (player_index + 1) % len(players)
    game_win_state = ttt.check_win()
    if print_board:
        print(game_win_state)
    return game_win_state


if __name__ == '__main__':
    N = 100000

    def run_games(players_list):
        random_random = [play_game(players_list, print_board=False) for _ in range(N)]
        p_x_win = 100 * np.mean([g == 'player x wins' for g in random_random])
        p_o_win = 100 * np.mean([g == 'player o wins' for g in random_random])
        draw = 100 * np.mean([g == 'draw' for g in random_random])
        if players_list[0].id is PlayerId.X:
            return p_x_win, p_o_win, draw
        else:
            return p_o_win, p_x_win, draw

    minmax_player_x = MinMaxPlayer(PlayerId.X)
    minmax_player_o = MinMaxPlayer(PlayerId.O)
    random_player_x = RandomPlayer(PlayerId.X)
    random_player_o = RandomPlayer(PlayerId.O)

    print('Player            | P1 Win | P2 Win  |  Draw')
    print('|:---|:---:|:---:|:---:|')
    p1_win, p2_win, draw = run_games([random_player_x, random_player_o])
    print(f'Random - Random   | {p1_win: 3.1f}% | {p2_win: 3.1f}%  | {draw: 3.1f}%')
    p1_win, p2_win, draw = run_games([minmax_player_x, random_player_o])
    print(f'MinMax - Random   | {p1_win: 3.1f}% | {p2_win: 3.1f}%  | {draw: 3.1f}%')
    p1_win, p2_win, draw = run_games([random_player_o, minmax_player_x])
    print(f'Random - MinMax   | {p1_win: 3.1f}% | {p2_win: 3.1f}%  | {draw: 3.1f}%')
    p1_win, p2_win, draw = run_games([random_player_x, minmax_player_o])
    print(f'Random - MinMax   | {p1_win: 3.1f}% | {p2_win: 3.1f}%  | {draw: 3.1f}%')
    p1_win, p2_win, draw = run_games([minmax_player_x, minmax_player_o])
    print(f'MinMax - MinMax   | {p1_win: 3.1f}% | {p2_win: 3.1f}%  | {draw: 3.1f}%')

    print('done')
