from typing import List
from environment import TicTacToe, PlayerId
from players import RandomPlayer


def play_game(players: List):
    # init game
    ttt = TicTacToe()

    # start game
    player_index = 0
    while ttt.check_win() == 'no win':
        ttt.player_move(players[player_index].id, players[player_index].next_move(ttt))
        ttt.print_board()
        player_index = (player_index + 1) % len(players)
    game_win_state = ttt.check_win()
    print(game_win_state)
    return game_win_state


if __name__ == '__main__':

    # init players
    players_list = [RandomPlayer(PlayerId.X), RandomPlayer(PlayerId.O)]
    win_state = play_game(players_list)
