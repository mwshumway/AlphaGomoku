"""
play_interactive.py: Play Gomoku interactively against the trained model

@author: Matt Shumway
"""

import numpy as np
import torch
from mcts import MCTS
from gomoku import GomokuEnv
from policy_value_network import PolicyValueNet


def play_interactive():
    """
    Play Gomoku interactively against the trained model
    """
    model = PolicyValueNet(board_size=6)
    model.load_model('/users/mwshumway/Downloads/model_last (2).pth')
    mcts = MCTS(model=model, n_playout=500)
    env = GomokuEnv(board_size=6, win_len=4)

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