"""
tests.py: Unit tests for AlphaGomoku

@author: Matt Shumway
"""

import pytest
from mcts import MCTS
from gomoku import GomokuEnv
from policy_value_network import PolicyValueNet

def test_action_validity(mcts, env):
    action, _ = mcts.get_action(env)
    is_valid_move = action in env.available_actions
    assert is_valid_move, f"Invalid move: {action}"
    print("Action validity test passed.")

def test_player_alternation(mcts, env, num_moves=5):
    while not env.done:
        action, _ = mcts.get_action(env)
        is_valid_move = action in env.available_actions
        assert is_valid_move, f"Invalid move: {action}"
        env.step(action)  # Apply move to the environment
        print(f"Player {-env.current_player} moved at {action}.")
        env.render()
    print("Player alternation test passed.")

def test_multiple_games(mcts, env, num_games=2):
    for _ in range(num_games):
        env.reset()
        mcts.reset()
        test_player_alternation(mcts, env)
    print("Multiple games test passed.")


if __name__ == "__main__":
    model = PolicyValueNet(board_size=3)
    mcts = MCTS(model=model, n_playout=100)
    env = GomokuEnv(board_size=3, win_len=3)

    # test_action_validity(mcts, env)
    # test_player_alternation(mcts, env)
    test_multiple_games(mcts, env)
    print("All tests passed.")
