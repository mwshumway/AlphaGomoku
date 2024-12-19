import numpy as np
import numpy as np
import torch
from mcts import MCTS
from gomoku import GomokuEnv
# from policy_value_network import PolicyValueNet
from policy_value_net_numpy import PolicyValueNetNumpy
import pickle

class RuleBasedAgent:
    """Simple rule-based agent for Gomoku."""
    def __init__(self, board_size, win_len):
        self.board_size = board_size
        self.win_len = win_len

    def choose_action(self, env):
        """
        Chooses an action based on the current board state.

        Args:
            env (GomokuEnv): The Gomoku environment.

        Returns:
            int: The chosen action (1D index).
        """
        board = env.get_state()
        player = env.current_player

        # Check for immediate win
        for action in env.available_actions:
            if self.is_winning_move(board, action, player):
                return action

        # Check for immediate block
        opponent = -player
        for action in env.available_actions:
            if self.is_winning_move(board, action, opponent):
                return action

        # Extend a line or create a threat
        for action in env.available_actions:
            if self.is_promising_move(board, action, player):
                return action

        # Default: pick a random valid move
        return np.random.choice(env.available_actions)

    def is_winning_move(self, board, action, player):
        """
        Check if placing a stone at the given action would result in a win for the player.

        Args:
            board (np.array): The current board state.
            action (int): The 1D index of the move to check.
            player (int): The player to check for (1 or -1).

        Returns:
            bool: True if the move results in a win, False otherwise.
        """
        temp_board = board.copy()
        row, col = divmod(action, self.board_size)
        temp_board[row, col] = player
        return self.check_winner(temp_board, row, col, player)

    def is_promising_move(self, board, action, player):
        """
        Check if placing a stone at the given action helps extend a line or creates a threat.

        Args:
            board (np.array): The current board state.
            action (int): The 1D index of the move to check.
            player (int): The player to check for (1 or -1).

        Returns:
            bool: True if the move is promising, False otherwise.
        """
        temp_board = board.copy()
        row, col = divmod(action, self.board_size)
        temp_board[row, col] = player
        return self.count_continuous_stones(temp_board, row, col, player) >= self.win_len - 2

    def count_continuous_stones(self, board, row, col, player):
        """
        Count the maximum number of continuous stones for the player in any direction.

        Args:
            board (np.array): The board state.
            row (int): Row index of the move.
            col (int): Column index of the move.
            player (int): The player to check for (1 or -1).

        Returns:
            int: The maximum count of continuous stones in any direction.
        """
        max_count = 0
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1  # Include the current stone
            for i in range(1, self.win_len):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and board[r, c] == player:
                    count += 1
                else:
                    break
            for i in range(1, self.win_len):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and board[r, c] == player:
                    count += 1
                else:
                    break
            max_count = max(max_count, count)
        return max_count

    def check_winner(self, board, row, col, player):
        """
        Check if placing a stone at (row, col) results in a win for the player.

        Args:
            board (np.array): The board state.
            row (int): Row index of the move.
            col (int): Column index of the move.
            player (int): The player to check for (1 or -1).

        Returns:
            bool: True if the move results in a win, False otherwise.
        """
        return self.count_continuous_stones(board, row, col, player) >= self.win_len
    

def play_one_game(mcts, model, rule_player, play_first=True, verbose=False):
    env = GomokuEnv(board_size=8, win_len=5)

    ai_turn = play_first

    while not env.done:
        if verbose:
            env.render()
        
        if ai_turn:
            action, _ = mcts.get_action(env)
        else:
            action = rule_player.choose_action(env)
        
        env.step(action)
        ai_turn = not ai_turn

    if env.winner == 1:
        if play_first:
            return 'win'
        else:
            return 'loss'
    elif env.winner == -1:
        if play_first:
            return 'loss'
        else:
            return 'win'
    else:
        return 'draw'
    
def main():
    net_params = pickle.load(open("best_policy_8_8_5.model", "rb"), encoding="bytes")
    model = PolicyValueNetNumpy(board_width=8, board_height=8, net_params=net_params)
    mcts = MCTS(model=model.policy_value_fn, n_playout=500)
    
    num_simulations = input("Enter the number of simulations to run: ")
    while not num_simulations.isdigit():
        print("Invalid input. Please enter a positive integer.")
        num_simulations = input("Enter the number of simulations to run: ")
    
    num_simulations = int(num_simulations)

    wins_first, losses_first, draws_first = 0, 0, 0
    wins_second, losses_second, draws_second = 0, 0, 0

    rule_player = RuleBasedAgent(board_size=8, win_len=5)

    for _ in range(num_simulations):
        res1 = play_one_game(mcts, model, rule_player, play_first=True)
        res2 = play_one_game(mcts, model, rule_player, play_first=False)

        if res1 == 'win':
            wins_first += 1
        elif res1 == 'loss':
            losses_first += 1
        else:
            draws_first += 1
        
        if res2 == 'win':
            wins_second += 1
        elif res2 == 'loss':
            losses_second += 1
        else:
            draws_second += 1
        
    print(f"AI plays first: {wins_first} wins, {losses_first} losses, {draws_first} draws")
    print()
    print(f"AI plays second: {wins_second} wins, {losses_second} losses, {draws_second} draws")


if __name__ == "__main__":
    main()
