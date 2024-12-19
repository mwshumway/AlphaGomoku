"""
vs_random.py: Play Gomoku interactively against a random agent

@author: Matt Shumway
"""

import numpy as np
import torch
from mcts import MCTS
from gomoku import GomokuEnv
# from policy_value_network import PolicyValueNet
from policy_value_net_numpy import PolicyValueNetNumpy
import pickle

def play_one_game(mcts, model, play_first=True, verbose=False):
    env = GomokuEnv(board_size=8, win_len=5)

    ai_turn = play_first

    while not env.done:
        if verbose:
            env.render()
        
        if ai_turn:
            action, _ = mcts.get_action(env)
        else:
            action = np.random.choice(env.available_actions)
        
        env.step(action)
        ai_turn = not ai_turn

    print(env.winner)

    
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

    for _ in range(num_simulations):
        res1 = play_one_game(mcts, model, play_first=True)
        res2 = play_one_game(mcts, model, play_first=False, verbose=True)

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