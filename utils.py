"""
utils.py: Utility functions for AlphaGomoku

@author: Matt Shumway
"""

import numpy as np
import torch

def softmax(x):
    """
    Compute the softmax of an array.
    
    Args:
        x (np.ndarray): The input array
    
    Returns:
        np.ndarray: The softmax of the input array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def prepare_single_input(board, player, last_move):
    """
    Prepares the model input for the policy-value network. 
    Encodes the board state from the perspective of the current player.

    Args:
        board (np.array): The game board, where:
                          1 represents player 1,
                         -1 represents player 2,
                          0 represents empty spaces.
        player (int): The current player's value (1 or -1).
        last_move (int or None): The index of the last move made, or None if no moves yet.
    
    Returns:
        torch.Tensor: A 4D tensor of shape (1, 5, board_height, board_width) as model input.
    """
    board = np.copy(board)


    square_state = np.zeros((4, board.shape[0], board.shape[1]))
    square_state[0][board == player] = 1  # Current player's stones
    square_state[1][board == -player] = 1  # Opponent's stones
    if last_move is not None:
        square_state[2][last_move // board.shape[1], last_move % board.shape[1]] = 1  # Last move
    square_state[3] = 1.0 if player == 1 else 0.0  # Current player

    return torch.from_numpy(square_state).float().unsqueeze(0)

def prepare_batch_data(play_data):
    """
    Prepares the model input for the policy-value network.
    Encodes the board states from the perspective of the current players.
    
    Args:
        play_data (list): A list of tuples (state, current_player, action_probs, action, winner).
    
    Returns:
        torch.Tensor: A 4D tensor of shape (batch_size, 5, board_height, board_width) as model input.
    """
    input_data, pi, z = [], [], []
    for state, current_player, action_probs, action, winner in play_data:
        input_data.append(prepare_single_input(state, current_player, action))
        pi.append(action_probs)
        z.append(winner)
    
    input_data = torch.cat(input_data, dim=0)  # Shape: (batch_size, 5, board_height, board_width) -- this is the model input
    pi = torch.from_numpy(np.array(pi, dtype=np.float32))  # Shape: (batch_size, board_size^2) -- action probabilities
    z = torch.from_numpy(np.array(z, dtype=np.float32))  # Shape: (batch_size,) -- winner
    return input_data, pi, z


    