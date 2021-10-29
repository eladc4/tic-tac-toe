import six, abc
from environment import PlayerId


@six.add_metaclass(abc.ABCMeta)
class BasePlayer:
    """
    Base player to define API.
    """
    def __init__(self, player_id: PlayerId):
        self.id = player_id

    def end_game(self, result):
        pass

    @abc.abstractmethod
    def next_move(self, board):
        """
        returns the player next move according to current board state
        :param board: board numpy array
        :return: index of next move
        """
