from MADQN import MADQN
from single_agent.utils_common import agg_double_list

import sys
import gym
import os
import numpy as np
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import highway_env
import visualization

os.environ["OFFSCREEN_RENDERING"] = "1"

MAX_EPISODES = 500000
EPISODES_BEFORE_TRAIN = 10
EVAL_EPISODES = 3
EVAL_INTERVAL = 100

# max steps in each episode, prevent from running too long
MAX_STEPS = 100

MEMORY_CAPACITY = 1000000
BATCH_SIZE = 512
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None

REWARD_DISCOUNTED_GAMMA = 0.99
EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 50000

def parse_args():
    """
    Description for this experiment:
        + easy: globalR
        + seed = 0
    """
    default_base_dir = "./results/"
    default_config_dir = 'configs/configs_ppo.ini'
    parser = argparse.ArgumentParser(description=('Train or evaluate policy on RL environment '
                                                  'using mappo'))
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--option', type=str, required=False,
                        default='train', help="train or evaluate")
    parser.add_argument('--config-dir', type=str, required=False,
                        default=default_config_dir, help="experiment config path")
    parser.add_argument('--model-dir', type=str, required=False,
                        default='', help="pretrained model path")
    parser.add_argument('--evaluation-seeds', type=str, required=False,
                        default=','.join([str(i) for i in range(0, 200, 20)]),
                        help="random seeds for evaluation, split by ,")
    args = parser.parse_args()
    return args

def run():
    train_dense = 1
    eval_dense = 2
    env = gym.make('merge-multi-agent-v0')
    env.config['traffic_density'] = train_dense
    env_eval = gym.make('merge-multi-agent-v0')
    env_eval.config['traffic_density'] = eval_dense
    state_dim = env.n_s
    action_dim = env.n_a
    print(f"state_dim:{state_dim}, action_dim:{action_dim}")
    print(f"dense{train_dense}_evaldense{eval_dense}")
    print(f"save to: rewardDQN_bsz{BATCH_SIZE}_gamma{REWARD_DISCOUNTED_GAMMA}_traindense{train_dense}_evaldense{eval_dense}.pdf")
    madqn = MADQN(env=env, memory_capacity=MEMORY_CAPACITY,
              state_dim=state_dim, action_dim=action_dim,
              batch_size=BATCH_SIZE, max_steps=MAX_STEPS,
              reward_gamma=REWARD_DISCOUNTED_GAMMA,
              epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
              epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
              episodes_before_train=EPISODES_BEFORE_TRAIN)
    video_logger = visualization.AnimationWrapper(save_floder='results')

    episodes = []
    eval_rewards = []
    eval_std = []
    while madqn.n_episodes < MAX_EPISODES:
        print("Episode", madqn.n_episodes)
        madqn.interact()
        frame = env.render(mode='rgb_array')
        video_logger.add_frame(frame)
        if madqn.n_episodes >= EPISODES_BEFORE_TRAIN:
            madqn.train()
        if madqn.episode_done and ((madqn.n_episodes + 1) % EVAL_INTERVAL == 0):
            rewards, _ = madqn.evaluation(env_eval, EVAL_EPISODES)
            rewards_mu, rewards_std = agg_double_list(rewards)
            print("Episode %d, Average Reward %.2f, Std %.2f" % (madqn.n_episodes + 1, rewards_mu, rewards_std))
            episodes.append(madqn.n_episodes + 1)
            eval_rewards.append(rewards_mu)
            eval_std.append(rewards_std)

    episodes = np.array(episodes)
    eval_rewards = np.array(eval_rewards)
    eval_std = np.array(eval_std)
    video_logger.save_video(file_name='video.gif')

    plt.figure()
    plt.plot(episodes, eval_rewards)
    plt.fill_between(episodes, eval_rewards-eval_std, eval_rewards+eval_std, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend(["DQN"])
    plt.savefig(f"rewardDQN_bsz{BATCH_SIZE}_gamma{REWARD_DISCOUNTED_GAMMA}_traindense{train_dense}_evaldense{eval_dense}.pdf")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        run(sys.argv[1])
    else:
        run()
