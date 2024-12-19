"""
policy_value_network.py: Optimized AlphaZero implementation for Gomoku

@author: Matt Shumway
"""

from gomoku import GomokuEnv
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torchsummary import summary
from graphviz import Digraph
import pickle
import numpy as np

class DepthwiseConv2d(nn.Module):
    """
    Depthwise separable convolution for efficient feature extraction.
    """
    def __init__(self, in_channels, out_channels):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block with depthwise separable convolutions and batch normalization.
    """
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = DepthwiseConv2d(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseConv2d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        y = self.bn2(self.conv2(y))
        return F.leaky_relu(x + y, negative_slope=0.01)
    


class PolicyValueNet(nn.Module):
    """
    Policy-value network model optimized for AlphaZero.
    """
    def __init__(self, board_size, n_res_blocks=3, channels=128, device='cpu'):
        super(PolicyValueNet, self).__init__()
        self.board_size = board_size
        self.n_res_blocks = n_res_blocks
        self.channels = channels
        self.device = device

        self.conv1 = nn.Conv2d(5, channels, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([ResBlock(channels, channels) for _ in range(self.n_res_blocks)])

        # policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # value head
        self.val_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_size * board_size, 32)
        self.val_fc2 = nn.Linear(32, 1)

        self.to(self.device)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        for block in self.res_blocks:
            x = block(x)

        # policy head
        p = F.leaky_relu(self.policy_conv(x), negative_slope=0.01)
        p = p.view(-1, 2 * self.board_size * self.board_size)
        p = F.log_softmax(self.policy_fc(p), dim=1)

        # value head
        v = F.leaky_relu(self.val_conv(x), negative_slope=0.01)
        v = v.view(-1, 2 * self.board_size * self.board_size)
        v = F.leaky_relu(self.val_fc1(v), negative_slope=0.01)
        v = torch.tanh(self.val_fc2(v))

        return p, v
    # def __init__(self, board_size):
    #     super(PolicyValueNet, self).__init__()

    #     self.board_size = board_size

    #     # common layers
    #     self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
    #     self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    #     self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    #     # action policy layers
    #     self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
    #     self.act_fc1 = nn.Linear(4*board_size*board_size,
    #                              board_size*board_size)
    #     # state value layers
    #     self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
    #     self.val_fc1 = nn.Linear(2*board_size*board_size, 64)
    #     self.val_fc2 = nn.Linear(64, 1)

    # def forward(self, state_input):
    #     # common layers
    #     x = F.relu(self.conv1(state_input))
    #     x = F.relu(self.conv2(x))
    #     x = F.relu(self.conv3(x))
    #     # action policy layers
    #     x_act = F.relu(self.act_conv1(x))
    #     x_act = x_act.view(-1, 4*self.board_size*self.board_size)
    #     x_act = F.log_softmax(self.act_fc1(x_act))
    #     # state value layers
    #     x_val = F.relu(self.val_conv1(x))
    #     x_val = x_val.view(-1, 2*self.board_size*self.board_size)
    #     x_val = F.relu(self.val_fc1(x_val))
    #     x_val = F.tanh(self.val_fc2(x_val))
    #     return x_act, x_val

    def get_params(self):
        """
        Get the parameters of the model.
        """
        return self.parameters()

    def save_model(self, filename):
        """
        Save the model to a file.
        """
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        """
        Load the model from a file.
        """
        self.load_state_dict(torch.load(filename, weights_only=True, map_location='cpu'))

def visualize():
    model = PolicyValueNet(board_size=15)  # Adjust based on your input

    # Use a dummy input with the correct shape (e.g., 1 for batch size)
    dummy_input = torch.randn(1, 5, 15, 15)  # (batch_size, channels, height, width)

    # Print a simple summary of the model
    summary(model, input_size=(5, 15, 15))  # Adjust input_size according to your network's input dimensions

def create_simple_graph():
    dot = Digraph()

    # Add layers
    dot.node('Input', 'Input (4, 15, 15)')
    dot.node('Conv1', 'Conv2d (4 -> 128, 3x3)')
    # dot.node('ResBlocks', 'Residual Blocks')
    dot.node('PolicyHead', 'Policy Conv & FC')
    dot.node('ValueHead', 'Value Conv & FC')
    dot.node('Output', 'Output (Policy & Value)')

    # Define the connections between layers
    dot.edge('Input', 'Conv1')
    dot.edge('Conv1', 'PolicyHead')
    dot.edge('Conv1', 'ValueHead')
    dot.edge('PolicyHead', 'Output')
    dot.edge('ValueHead', 'Output')

    # Render the graph (will generate a PNG image)
    dot.render('simple_nn_graph', format='png', cleanup=True)

    # save the graph locally
    dot.save('simple_nn_graph')

if __name__ == '__main__':
    # visualize()
    create_simple_graph()