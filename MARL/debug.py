# import highway_env
# import gym
#
# os.environ["SDL_VIDEODRIVER"] = "dummy"
#
# env = gym.make("parking-multi-agent-v0")
# env.reset(num_CAV=4)
#
# for t in range(20):
#     # env.reset()
#     obs, reward, terminal, info = env.step(env.action_space.sample())
#
#     o=env.render(mode='rgb_array')
#     print(t)


import gym
from IPython import display
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import os
import sys
import highway_env
import numpy as np
# os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["OFFSCREEN_RENDERING"] = "1"

env = gym.make("parking-multi-agent-v0")
env.reset(num_CAV=4)
plt.figure(figsize=(9,9))
img = plt.imshow(env.render(mode='rgb_array')) # only call this once
plt.show()
for _ in range(100):
    # os.environ["SDL_VIDEODRIVER"] = None
    # os.environ["OFFSCREEN_RENDERING"] = "0"
    # env.render(mode='human')

    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    obs = env.render(mode='rgb_array')
    img.set_data(obs) # just update the data
    # display.display(plt.gcf())
    # display.clear_output(wait=True)

    agent_obs, reward, terminal, info = env.step(env.action_space.sample())
    # plt.show()
env.close()