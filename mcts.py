"""
mcts.py: Monte Carlo Tree Search implementation for Gomoku

@author: Matt Shumway
"""

import numpy as np
import copy

def softmax(x):
    """
    Compute the softmax of a vector.
    """
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class Node:
    """
    A node in the MCTS tree.
    """

    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.Q = 0
        self.P = prior_p
        self.ucb = 0

    def get_ucb(self, c_puct):
        """
        Compute the UCB score for this node.
        """
        return self.Q + c_puct * self.P * np.sqrt(self.parent.n_visits) / (1 + self.n_visits)

    def expand(self, action_priors, legal_actions):
        """
        Expand tree by creating new children for each possible action.
        """
        for action, prob in enumerate(action_priors):
            if action in legal_actions and action not in self.children:
                self.children[action] = Node(self, prob)

    def select(self, c_puct):
        """
        Select action among children that gives maximum UCB value.
        """
        for _, node in self.children.items():
            node.ucb = node.get_ucb(c_puct)
        return max(self.children.items(), key=lambda x: x[1].ucb)

    def update(self, value):
        """
        Update node values from leaf evaluation.
        """
        self.n_visits += 1
        self.Q += (value - self.Q) / self.n_visits  # Incremental update for average Q

    def backpropagate(self, value):
        """
        Update node values from leaf evaluation, backpropagating to the root.
        """
        if self.parent:
            self.parent.backpropagate(-value)
        self.update(value)

    def is_leaf(self):
        """
        Check if the node is a leaf.
        """
        return len(self.children) == 0


class MCTS:
    """
    Monte Carlo Tree Search.
    """

    def __init__(self, model, c_puct=5, n_playout=10000):
        self.model = model
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.root = Node()

    def reset(self):
        """
        Reset the tree.
        """
        self.root = Node()

    def get_action(self, env, temp=1e-3, is_selfplay=False):
        """
        Get the best action from the current state.
        """
        for _ in range(self.n_playout):
            env_copy = copy.deepcopy(env)
            self.playout(env_copy)

        visit_counts = [(action, node.n_visits) for action, node in self.root.children.items()]
        actions, counts = zip(*visit_counts)
        action_probs = softmax(1.0 / temp * np.log(np.array(counts) + 1e-10))

        move_probs = np.zeros(env.board_size ** 2)
        move_probs[list(actions)] = action_probs

        if is_selfplay:
            action = np.random.choice(
                actions, p=0.75 * action_probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(action_probs)))
            )
        else:
            action = np.random.choice(actions, p=action_probs)

        if is_selfplay:
            self.update_with_move(action)
        else:
            self.update_with_move(-1)

        return action, move_probs

    def playout(self, env):
        """
        Run a single playout from the root to the leaf, getting a value at the leaf and backpropagating it.
        """
        node = self.root
        action = None

        while not node.is_leaf():
            action, node = node.select(self.c_puct)
            if action not in env.available_actions:
                raise ValueError("Illegal action selected")
            env.step(action)

        # Normalize the board state for the current player's perspective
        state = env.get_encoded_state()
        player = env.current_player

        # Evaluate the leaf using the model
        action_probs, value = self.model(state, env.available_actions)

        # action probs is a list[(action, prob), ...]
        # Convert this to an array of probabilities
        legal_action_probs = np.zeros(env.board_size ** 2)
        for action, prob in action_probs:
            legal_action_probs[action] = prob

        legal_action_probs /= (np.sum(legal_action_probs) + 1e-8)

        # # Adjust action probabilities for legal moves
        # available_actions = env.available_actions
        # legal_action_probs = np.zeros(env.board_size ** 2)
        # legal_action_probs[available_actions] = action_probs[available_actions]
        # legal_action_probs /= (np.sum(legal_action_probs) + 1e-8)

        if env.done:
            value = 1 if env.winner == player else -1 if env.winner == -player else 0
        else:
            node.expand(legal_action_probs, env.available_actions)

        # Backpropagate the value, flipping it for the opponent's perspective
        node.backpropagate(value * player)

    def update_with_move(self, last_move):
        """
        Update the tree with the last move and set the new root.
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = Node()
