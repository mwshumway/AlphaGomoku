"""game.py: A class for the game of Gomoku.

Matt Shumway"""

import numpy as np

class GameBoard():
    def __init__(self, **kwargs):
        """
        Initialize the game board.
        
        :param kwargs: The keyword arguments for the game board.
        """
        self.size = kwargs.get('size', 8)
        self.winning_length = kwargs.get('winning_length', 5)
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.player = 1  # 1: black, 2: while
        self.winner = 0  # 0: no winner, 1: black, 2: white
        self.done = False
        self.valid_actions = [(i, j) for i in range(self.size) for j in range(self.size)]
        self.last_move = None
    
    def reset(self):
        """
        Reset the game board
        
        :return: The initial state of the game board.
        """
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.player = 1
        self.winner = 0
        self.done = False
        return self.get_state()
    
    def get_state(self):
        """
        Get the current state of the game board.
        
        :return: The current state of the game board.
        """
        return np.copy(self.board)
    
    def step(self, action):
        """
        Take a step in the game.
        
        :param action: The action to take.
        :return: The next state, reward, done, and info.
        """
        if action not in self.valid_actions:
            raise ValueError(f"Invalid action: {action}")
        
        self.last_move = (self.player, action)  # store the last move for encoding
        
        self.valid_actions.remove(action)
        row, col = action
        self.board[row, col] = self.player
        if self.check_win(row, col):
            self.winner = self.player
            self.done = True
            reward = 1
        elif np.all(self.board != 0):
            self.done = True
            self.winner = 0
            reward = 0
        else:
            reward = 0
            self.switch_player()
        return self.get_state(), reward, self.done, {}
    
    def switch_player(self):
        """
        Switch the player.
        """
        self.player = 1 if self.player == 2 else 2

    def check_win(self, row, col):
        """
        Check if the player has won.
        
        :param row: The row of the last move.
        :param col: The column of the last move.
        :return: True if the player has won, False otherwise.
        """
        return self.check_horizontal(row, col) or self.check_vertical(row, col) or self.check_diagonal(row, col)

    def check_horizontal(self, row, col):
        """
        Check if the player has won horizontally.
        
        :param row: The row of the last move.
        :param col: The column of the last move.
        :return: True if the player has won horizontally, False otherwise.
        """
        c = 0
        for j in range(self.size):
            if self.board[row, j] == self.player:
                c += 1
                if c == self.winning_length:
                    return True
            else:
                c = 0
        return False
    
    def check_vertical(self, row, col):
        """
        Check if the player has won vertically.
        
        :param row: The row of the last move.
        :param col: The column of the last move.
        :return: True if the player has won vertically, False otherwise.
        """
        c = 0
        for i in range(self.size):
            if self.board[i, col] == self.player:
                c += 1
                if c == self.winning_length:
                    return True
            else:
                c = 0
        return False
    
    def check_diagonal(self, row, col):
        """
        Check if the player has won diagonally.
        
        :param row: The row of the last move.
        :param col: The column of the last move.
        :return: True if the player has won diagonally, False otherwise.
        """
        c = 0
        # Check diagonal from top-left to bottom-right
        for i in range(-self.winning_length + 1, self.winning_length):
            if 0 <= row + i < self.size and 0 <= col + i < self.size:
                if self.board[row + i, col + i] == self.player:
                    c += 1
                    if c == self.winning_length:
                        return True
                else:
                    c = 0
        # Check diagonal from top-right to bottom-left
        c = 0
        for i in range(-self.winning_length + 1, self.winning_length):
            if 0 <= row + i < self.size and 0 <= col - i < self.size:
                if self.board[row + i, col - i] == self.player:
                    c += 1
                    if c == self.winning_length:
                        return True
                else:
                    c = 0
        return False
    
    def encode_state(self):
        """
        Encode the state of the game board. 
        Encode the state of the game board as 4 channels:
        1. Player 1 pieces
        2. Player 2 pieces
        3. The current turn
        4. The last move
        
        :return: The encoded state of the game board.
        """
        channel1 = np.where(self.board == 1, 1, 0)
        channel2 = np.where(self.board == 2, 1, 0)
        channel3 = np.full((self.size, self.size), self.player)
        channel4 = np.zeros((self.size, self.size))
        if self.last_move is not None:
            player, (row, col) = self.last_move
            channel4[row, col] = player
        
        return np.stack([channel1, channel2, channel3, channel4], axis=2)
    

class GameSimulator():
    def __init__(self, **kwargs):
        """
        Initialize the game simulator.
        
        :param kwargs: The keyword arguments for the game simulator.
        """
        self.board_size = kwargs.get('board_size', 8)
        self.winning_length = kwargs.get('winning_length', 5)
        self.board = GameBoard(size=self.board_size, winning_length=self.winning_length)
    
    def draw_board(self):
        """
        Draw the game board.
        """
        symbols = {0: '.', 1: 'X', 2: 'O'}
        print('   ' + ' '.join([f'{i:2}' for i in range(self.board_size)]))
        for row_index, row in enumerate(self.board.board):
            row_str = ' '.join([symbols[cell] for cell in row])
            print(f'{row_index:2} {row_str}')
    
    def play_game(self, player1, player2):
        """
        Play a game between 2 players.
        
        :param player1: The first player.
        :param player2: The second player.
        
        :return: The winner of the game.
        """
        self.board.reset()
        while not self.board.done:
            if self.board.player == 1:
                action = player1.get_action(self.board.get_state())
            else:
                action = player2.get_action(self.board.get_state())
            next_state, reward, done, _ = self.board.step(action)
        return self.board.winner
    
    def self_play(self, player):
        """
        Have a player play a game against itself.
        
        :param player: The player to play against itself.
        
        :return: The winner of the game.
        """
        self.board.reset()
        while not self.board.done:
            action = player.get_action(self.board.get_state())
            next_state, reward, done, _ = self.board.step(action)
        return self.board.winner

