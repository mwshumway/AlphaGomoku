import numpy as np

class GomokuSimulator:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)  # 0: empty, 1: black, 2: white
        self.player = 1  # 1: black, 2: white
        self.winner = 0  # 0: no winner, 1: black, 2: white
        self.done = False   
        self.valid_actions = [(i, j) for i in range(board_size) for j in range(board_size)]
    
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.player = 1
        self.winner = 0
        self.done = False
        return self.get_state()
    
    def get_state(self):
        return np.copy(self.board)

    def step(self, action):
        
        self.valid_actions.remove(action)

        row, col = action  

        # Don't need because we are only selecting valid actions from the action space.
        # if self.board[row, col] != 0 or self.done:
        #     return self.get_state(), -1, True, {'msg': 'Invalid move'}
        
        self.board[row, col] = self.player

        if self.check_win(row, col):
            self.winner = self.player
            self.done = True
            reward = 1
        elif np.all(self.board != 0):  # Board is full
            self.done = True
            self.winner = 0
            reward = 0
        else:
            reward = 0  # No winner yet
            self.switch_player()

        return self.get_state(), reward, self.done, {}

    def switch_player(self):
        self.player = 1 if self.player == 2 else 2

    def check_horizontal(self, row, col):
        c = 0
        for j in range(self.board_size):
            if self.board[row, j] == self.player:
                c += 1
                if c == 5:
                    return True
            else:
                c = 0
        return False
    
    def check_vertical(self, row, col):
        c = 0
        for i in range(self.board_size):
            if self.board[i, col] == self.player:
                c += 1
                if c == 5:
                    return True
            else:
                c = 0
        return False

    def check_diagonal(self, row, col):
        c = 0
        # Check diagonal from top-left to bottom-right
        for i in range(-4, 5):
            if 0 <= row + i < self.board_size and 0 <= col + i < self.board_size:
                if self.board[row + i, col + i] == self.player:
                    c += 1
                    if c == 5:
                        return True
                else:
                    c = 0
        # Check diagonal from top-right to bottom-left
        c = 0
        for i in range(-4, 5):
            if 0 <= row + i < self.board_size and 0 <= col - i < self.board_size:
                if self.board[row + i, col - i] == self.player:
                    c += 1
                    if c == 5:
                        return True
                else:
                    c = 0
        return False


    def check_win(self, row, col):
        """Check if the player just won with their most recent move."""
        if self.check_horizontal(row, col) or self.check_vertical(row, col) or self.check_diagonal(row, col):
            return True
        return False


