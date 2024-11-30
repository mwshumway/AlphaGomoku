from gomoku_simulator import GomokuSimulator
import random


board_size = 10
env = GomokuSimulator(board_size=board_size)
state = env.reset()
done = False
while not done:
    action = random.choice(env.valid_actions)
    state, reward, done, info = env.step(action)
    if done:
        print(f"Winner: {env.winner}")
        print(state)

