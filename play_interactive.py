"""
play_interactive.py: Play Gomoku interactively against the trained model

@author: Matt Shumway
"""

import numpy as np
import torch
from mcts import MCTS
from gomoku import GomokuEnv
# from policy_value_network import PolicyValueNet
from policy_value_net_numpy import PolicyValueNetNumpy
import pickle


def play_interactive():
    """
    Play Gomoku interactively against the trained model
    """
    # Load in the trained weights
    net_params = pickle.load(open("best_policy_8_8_5.model", "rb"), encoding="bytes")

    # Initialize the model
    model = PolicyValueNetNumpy(board_width=8, board_height=8, net_params=net_params)

    total_params = sum(param.size for param in model.params)

    print(f"Total number of parameters in the model: {total_params}")

    mcts = MCTS(model=model.policy_value_fn, n_playout=500)
    env = GomokuEnv(board_size=8, win_len=5)

    while not env.done:
        env.render()
        if env.current_player == 1:
            action = input("Enter your move: ")
            # Convert row, col to index
            row, col = map(int, action.split(","))
            action = row * env.board_size + col
            while action not in env.available_actions:
                print("Invalid move. Please try again.")
                action = input("Enter your move: ")
                row, col = map(int, action.split(","))
                action = row * env.board_size + col
        else:
            action, _ = mcts.get_action(env)
        env.step(action)

    env.render()
    if env.winner == 1:
        print("You win!")
    elif env.winner == -1:
        print("You lose!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    play_interactive()