"""
gomoku.py: Gomoku game implementation

@author: Matt Shumway
"""

import numpy as np
import gym
from gym import spaces

class GomokuEnv(gym.Env):
    """Gomoku game environment built on OpenAI Gym"""

    def __init__(self, board_size=15, win_len=5):
        super(GomokuEnv, self).__init__()
        self.board_size = board_size
        self.win_len = win_len
        self.action_space = spaces.Discrete(board_size ** 2)  # one action for each board position
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(board_size, board_size), dtype=np.int8
            )  # 0 for empty, 1 for player 1, -1 for player 2
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.available_actions = list(range(board_size ** 2))
        self.winner = 0

        self.states = {}
        self.last_move = None

    def reset(self):
        """
        Resets the board to its initial state and returns the board
        """
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.available_actions = list(range(self.board_size ** 2))
        self.winner = 0
    
    def step(self, action):
        """
        Executes a move on the board.
        
        Args:
            action (int): The cell index (flattened) where the current player places their stone.
        
        Returns:
            observation (np.array): The current board state.
            reward (float): The reward for the current move.
            done (bool): Whether the game has ended.
            info (dict): Additional information.
        """
        if self.done:
            raise ValueError("Game is over. Call 'reset' to start a new game.")
        
        row, col = divmod(action, self.board_size)  # convert from 1D to 2D index
        if self.board[row, col] != 0:
            raise ValueError("Invalid move. Cell already occupied.")
        
        self.board[row, col] = self.current_player
        self.available_actions.remove(action)
        self.states[action] = self.current_player
        self.last_move = action

        if self.check_winner(row, col):
            reward = 1 if self.current_player == 1 else -1
            self.done = True
        elif np.all(self.board != 0):
            reward = 0
            self.done = True
        else:
            reward = 0
        
        self.current_player = -self.current_player  # switch players

        return self.board, reward, self.done, {}
    
    def check_winner(self, row, col):
        """
        Checks if the current player has won the game.
        
        Args:
            row (int): The row index of the last move.
            col (int): The column index of the last move.
            
        Returns:
            bool: True if the current player has won, False otherwise.
        """
        player = self.board[row, col]
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for i in range(1, self.win_len):
                r, c = row + dr * i, col + dc * i
                if r < 0 or r >= self.board_size or c < 0 or c >= self.board_size or self.board[r, c] != player:
                    break
                count += 1
            for i in range(1, self.win_len):
                r, c = row - dr * i, col - dc * i
                if r < 0 or r >= self.board_size or c < 0 or c >= self.board_size or self.board[r, c] != player:
                    break
                count += 1
            if count >= self.win_len:
                self.winner = player
                return True
        return False
    
    def render(self, mode='human'):
        """
        Renders the current board state.
        
        Args:
            mode (str): The rendering mode.
        """
        print(" ", end=" ")
        for idx in range(self.board_size):
            print(idx, end=" ")
        print()
        for idx, row in enumerate(self.board):
            print(idx, end=" ")
            print(" ".join(["X" if cell == 1 else "O" if cell == -1 else "." for cell in row]))
        print()
    
    def close(self):
        """
        No resources to clean up in this simple environment.
        """
        pass

    def get_state(self):
        """
        Returns the current board state.
        
        Returns:
            np.array: The current board state.
        """
        return self.board
    
    def legal_moves_mask(self):
        """
        Returns a mask of legal moves.
        
        Returns:
            np.array: A mask of legal moves.
        """
        mask = np.zeros(self.board_size ** 2, dtype=np.bool)
        mask[self.available_actions] = True
        return mask
    
    def get_encoded_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.board_size, self.board_size))

        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.board_size,
                            move_curr % self.board_size] = 1.0
            square_state[1][move_oppo // self.board_size,
                            move_oppo % self.board_size] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.board_size,
                            self.last_move % self.board_size] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]


if __name__ == "__main__":
    # Instantiate the environment
    env = GomokuEnv(board_size=3, win_len=3)

    # Reset the environment
    state = env.reset()
    env.render()

    done = False

    while not done:
        # Sample a random action
        action = env.action_space.sample()
        try:
            # Perform the action
            state, reward, done, info = env.step(action)
            env.render()
        except ValueError as e:
            # Handle invalid moves
            print(e)

    print("Game Over!")
    print("Reward:", reward)

        


    