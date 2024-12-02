"""Training Algorithm of a DQN:
(taken from original DQN paper)

Initialize replay memory D to to capacity N
Initialize action-value function Q with random weights
For episode = 1, M DO
    initialize sequence s_1 = {x_1} and preprocessed sequences \phi_1 = \phi(s_1)
    for t = 1, T DO
        with P(\eps) select a random action a_t
        otherwise select a_t = max_a Q(\phi(s_t), a; \theta) 
        Execute action a_t in emulator and observe reward r_t and image x_{t+1}
        Set s_{t+1} = s_t, a_t, x_{t+1} and preprocess \phi_{t+1} = \phi(s_{t+1})
        Store transition (\phi_t, a_t, r_t, \phi_{t+1} in D)
        Sample random minibatch of transitions from D
        Set y_j = r_j for terminal \phi_{j+1}
                  r_j + \gamma max_a' Q(\phi_{j+1}, a'; \theta) for nonterminal
        Perform a gradient descent step on (y_j - Q(\phi_j, a_j; \theta))^2
    end for
end for
"""
from collections import deque
import random


class ReplayMemory():
    def __init__(self, memory_size, batch_size):
        """
        Initialize the replay memory.
        
        :param memory_size: The maximum size of the replay memory.
        :param batch_size: The number of experiences to sample from memory when training.
        """
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
    
    def add(self, experience):
        """
        Add a new experience to memory.
        
        :param experience: A tuple (state, action, reward, next_state, done).
        """
        self.memory.append(experience)
    
    def sample(self, experience):
        """
        Randomly sample a batch of experiences from the memory.
        
        :return: A batch of experiences as a list.
        """
        return random.sample(self.memory, self.batch_size)
    
    def __len__(self):
        """
        Return the current size of the memory.
        
        :return: num experiences stored.
        """
        return len(self.memory)



def dqn_algorithm(opt):
    """Implementation of the DQN algorithm."""
    memory = ReplayMemory(opt.memory_size, opt.batch_size)
    Q = DQN()