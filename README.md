# AlphaGomoku: Applying AlphaZero to the Game of Gomoku

## Overview
This project adapts the **AlphaZero algorithm** to Gomoku, a two-player strategy game played on a 15x15 board. The goal of Gomoku is to align five stones of the same color in a row—horizontally, vertically, or diagonally. While deceptively simple, Gomoku features strategic depth and serves as an excellent testbed for reinforcement learning (RL) techniques.

The AlphaZero algorithm combines a policy-value neural network, Monte Carlo Tree Search (MCTS), and self-play to achieve superhuman performance in various games. This implementation explores how well AlphaZero performs in Gomoku and highlights challenges in replicating AlphaZero with limited computational resources.

---

## Table of Contents
- [Introduction](#introduction)
- [Related Work](#related-work)
- [AlphaZero Algorithm](#alphazero-algorithm)
  - [Policy-Value Network](#policy-value-network)
  - [Monte Carlo Tree Search (MCTS)](#monte-carlo-tree-search-mcts)
  - [Self-Play](#self-play)
- [Methodology](#methodology)
- [References](#references)

---

## Introduction
Reinforcement learning (RL) is a powerful paradigm for training agents to interact with environments in ways that maximize performance based on pre-defined reward signals or costs. Over the past decade, advancements in innovative algorithms and computational capabilities have revolutionized RL, enabling efficient exploration of vast state and action spaces and the accurate approximation of value functions and policies.

One notable breakthrough was the development of an algorithm designed for the ancient Chinese game of Go. This algorithm, known as **AlphaGo Zero**, achieved superhuman performance without relying on human knowledge. The methodology was later generalized into the **AlphaZero algorithm**, which has mastered a variety of environments and demonstrated utility in applications like discovering efficient matrix multiplication algorithms.

This project provides an in-depth description of the AlphaZero algorithm and adapts it to Gomoku. While Gomoku strategies are more approachable than games like Go or Chess, its complexity makes it a valuable benchmark for RL algorithms. Notably, Gomoku is a theoretically solved game, where the starting player is guaranteed to win under perfect play.

---

## Related Work
Many others have explored similar projects applying AlphaZero to Gomoku. Examples include:
- [Liang et al. (2023)](https://example.com) and other open-source implementations.
- Projects with access to larger GPU or CPU clusters for optimized agents.

While this project may not break new ground, it has been a highly informative and enjoyable experience for the author.

---

## AlphaZero Algorithm

### Policy-Value Network
The AlphaZero algorithm uses a **policy-value network** parameterized by weights \( \theta \). This network outputs:
1. \( p_s \): A probability distribution over the action space \( A \), approximating the probability of optimal actions.
2. \( v_s \): A scalar representing the expected reward for being in a given state.

In Gomoku, states are represented as matrices modeling the game board:
- `0`: Empty square
- `1`: Player 1’s stone
- `2`: Player 2’s stone

To account for spatial patterns (e.g., rows, blocks, threats), the neural network includes convolutional layers.  
**[Insert NN architecture diagram here]**

---

### Monte Carlo Tree Search (MCTS)
MCTS intelligently simulates actions while balancing exploration and exploitation using the policy-value network. The search process involves four steps:

1. **Selection**  
   Traverse the tree using the PUCT formula:  
   \[
   a^* = \arg\max_a \left( Q(s, a) + c \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)} \right)
   \]  

2. **Expansion**  
   Add child nodes for all possible actions and initialize them with prior probabilities \( P(s, a) \) from the policy network.

3. **Simulation**  
   Use \( v_s \) (predicted by the policy-value network) as the evaluation of the leaf node.

4. **Backpropagation**  
   Update \( Q(s, a) \) (expected value) and \( N(s, a) \) (visit count) for all nodes on the path.

After all simulations, the normalized visit counts \( \pi(a) \propto N(s, a) \) form the output policy distribution. This balances exploration and exploitation for optimal decision-making.

---

### Self-Play
The self-play phase generates training data by having the agent play games against itself. At each step:
- The state, MCTS action probabilities, and rewards are recorded.
- This data is used to iteratively train the neural network.

The process continues until a stopping condition (e.g., a set number of training iterations) is reached.

---

## Methodology
**[Insert detailed methodology for implementing AlphaZero in Gomoku here, including training setup and evaluation metrics.]**

---

## References
- AlphaGo Zero: [Silver et al., 2017](https://example.com)
- Efficient matrix multiplication: [Fawzi et al., 2022](https://example.com)
- Solving Gomoku: [Allis et al., 1993](https://example.com)
- Related implementations: [Liang et al., 2023](https://example.com)

---

## Future Work
- Experimenting with larger board sizes or varying rules.
- Optimizing hyperparameters for better performance.
- Leveraging distributed computing to speed up training.
