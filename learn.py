import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from alphagomoku.gomoku import GomokuEnv
from mcts import MCTS
from policy_value_network import PolicyValueNet
from utils import *
import yaml
import tqdm
import random
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def add(self, data):
        if len(self.buffer) + len(data) > self.capacity:
            self.buffer = self.buffer[len(data):]
        self.buffer.extend(data)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)



def learn(config):
    """
    Train the AlphaGomoku model. Self-play and training are interleaved.
    Args:
        config (dict): The configuration dictionary
    """
    # Load the configuration
    board_size = config["board_size"]
    win_len = config["win_len"]
    n_playout = config["n_playout"]
    c_puct = config["c_puct"]
    lr = config["lr"]
    n_epochs = config["n_epochs"]
    batch_size = config["batch_size"]
    n_self_play_games = config["n_self_play"]
    log_interval = config["log_interval"]
    replay_capacity = config["replay_capacity"]

    # Initialize model
    model = PolicyValueNet(board_size=board_size)
    
    # Save the initial model for evaluation
    model.save_model('model_last.pth')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    # Initialize MCTS and Replay Buffer
    mcts = MCTS(model=model, c_puct=c_puct, n_playout=n_playout)
    replay_buffer = ReplayBuffer(capacity=replay_capacity)

    # Initialize environment
    env = GomokuEnv(board_size=board_size, win_len=win_len)

    # Training loop
    loop = tqdm.tqdm(range(n_epochs))

    # logs
    win_ratio_list = []
    train_loss_list = []

    for epoch in loop:
        # Adjust temperature based on epoch
        temp = max(1.0 - epoch / 10, 0.1)

        # Collect self-play data
        print("Collecting self-play data...")
        play_data = self_play(env, mcts, temp=temp, n_games=n_self_play_games)
        replay_buffer.add(play_data)

        # Sample data from the replay buffer
        sampled_data = replay_buffer.sample(batch_size)
        states, probs, rewards = prepare_batch_data(sampled_data)

        # Train the model
        print("Training the model...")
        train_loss = train(model, optimizer, states, probs, rewards, batch_size=batch_size, n_epochs=1)
        train_loss_list.append(np.mean(train_loss))
        scheduler.step()
        print('trained model for epoch', epoch)

        # Evaluate the model
        if epoch % log_interval == 0:
            print("Evaluating the model...")
            win_ratio = evaluate(model, n_games=5, board_size=board_size, win_len=win_len)
            win_ratio_list.append(win_ratio)
            loop.set_description(f"Win ratio: {win_ratio:.3f}")

            # Save the model
            model_file = f"model_last.pth"
            model.save_model(model_file)

    # Save the final model
    model_file = f"model_last.pth"

    # Plot the training loss and win ratio
    plt.figure()
    plt.plot(train_loss_list)
    plt.xlabel("Epoch (cycle of self-play and training)")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("train_loss.png")

    plt.figure()
    plt.plot(win_ratio_list)
    plt.xlabel(f"Evalution Epoch, Evaluated every {log_interval} epochs")
    plt.ylabel("Win Ratio")
    plt.title("Win Ratio")
    plt.savefig("win_ratio.png")


def evaluate(model, n_games=10, board_size=6, win_len=4):
    """
    Evaluate the model by playing games against an older version of itself.
    Args:
        model (PolicyValueNet): The model to evaluate.
        n_games (int): The number of games to play.
        board_size (int): The size of the board.
        win_len (int): The number of consecutive stones needed to win.
    Returns:
        float: The win ratio of the model.
    """
    win_count = 0
    play_first = True
    old_model = PolicyValueNet(board_size=board_size)
    old_model.load_model("model_last.pth")  # Load the previous best model

    for _ in range(n_games):
        env = GomokuEnv(board_size=board_size, win_len=win_len)
        mcts_new = MCTS(model=model, c_puct=5, n_playout=500)
        mcts_old = MCTS(model=old_model, c_puct=5, n_playout=500)
        winner = play_game_between_agents(env, mcts_new, mcts_old, play_first=play_first)
        play_first = not play_first
        if winner == 1:  # Model wins
            win_count += 1

    return win_count / n_games


def play_game_between_agents(env, mcts1, mcts2, play_first=True):
    """
    Play a game between two agents.
    Args:
        env (GomokuEnv): The game environment.
        mcts1, mcts2 (MCTS): The MCTS objects for each agent.
        play_first (bool): Whether the first model should play first.
    Returns:
        int: 1 if mcts1 wins, -1 if mcts2 wins.
    """
    env.reset()
    done = False

    while not done:
        if play_first:
            action, _ = mcts1.get_action(env, temp=1e-3, is_selfplay=False)
            play_first = False
        else:
            action, _ = mcts2.get_action(env, temp=1e-3, is_selfplay=False)
            play_first = True
        _, _, done, _ = env.step(action)

    return env.winner


def self_play(env, mcts, temp=1.0, n_games=1):
    """
    Run self-play games using the MCTS player.
    
    Args:
        env (GomokuEnv): The game environment.
        mcts (MCTS): The Monte Carlo Tree Search object.
        temp (float): The temperature parameter.
        n_games (int): The number of games to play.
    
    Returns:
        list: The self-play data. Each element is a tuple of (state, current_player, action_probs, action, winner).
    """
    play_data = []
    for _ in range(n_games):
        env.reset()
        mcts.reset()
        state = env.get_state()
        game_data = []

        while True:
            current_player = env.current_player
            action, action_probs = mcts.get_action(env, temp=temp, is_selfplay=True)
            game_data.append((state, current_player, action_probs, action))
            state, _, done, _ = env.step(action)
            current_player = env.current_player
            if done:
                break
        
        winner = env.winner
        game_data = [(state, current_player, action_probs, action, winner) for state, current_player, action_probs, action in game_data]

        play_data.extend(game_data)
    
    return play_data
        
    
def train(model, optimizer, state_batch, mcts_probs_batch, winner_batch, batch_size, n_epochs):
    """
    Train the model using the self-play data.
    
    Args:
        model (PolicyValueNet): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        state_batch (np.array): The batch of states.
        mcts_probs_batch (np.array): The batch of action probabilities.
        winner_batch (np.array): The batch of winners.
        batch_size (int): The batch size.
        n_learn_steps (int): The number of training steps.
    
    Returns:
        list: The training loss.
    """
    train_loss = []
    for i in range(n_epochs):
        # Sample a batch of data
        indices = np.random.choice(range(len(state_batch)), batch_size)
        state_batch = state_batch[indices]
        mcts_probs_batch = mcts_probs_batch[indices]
        winner_batch = winner_batch[indices]

        optimizer.zero_grad()
        action_probs, value = model(state_batch)

        # Calculate the loss
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.sum(mcts_probs_batch * action_probs) / batch_size  # action probs are already log probs
        loss = value_loss + policy_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Loss: {loss.item():.3f}")
            print(f"Value loss: {value_loss.item():.3f}")
            print(f"Policy loss: {policy_loss.item():.3f}")
            print()
        
        train_loss.append(loss.item())
    
    return train_loss


if __name__ == "__main__":
    with open("config.yml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    learn(config)

