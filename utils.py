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

    # Layers encoding the board from the current player's perspective
    self_player = np.where(board == player, 1, 0)      # Current player's stones
    opponent_player = np.where(board == -player, 1, 0) # Opponent's stones
    empty = np.where(board == 0, 1, 0)                 # Empty spaces

    # Layer for the current player (constant layer of the current player's ID)
    current_player_layer = np.full_like(board, player)

    # Layer for the last move
    last_move_layer = np.zeros_like(board)
    if last_move is not None:
        row, col = divmod(last_move, board.shape[1])  # Convert index to row, col
        last_move_layer[row, col] = 1  # Mark the last move location

    # Stack layers into a single tensor
    input_data = np.stack([self_player, opponent_player, empty, current_player_layer, last_move_layer])

    # Convert to a PyTorch tensor and add a batch dimension
    return torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 5, board_height, board_width)


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


    