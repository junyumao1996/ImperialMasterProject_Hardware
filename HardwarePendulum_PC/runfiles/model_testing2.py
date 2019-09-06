import sys
sys.path.append('/Users/indurance/PycharmProjects/MasterProject')
import gym
import tensorflow as tf
import time
from envs.real_circular_pendulum import RealCircularPendulumEnv
from agents.DDPG import DDPG
from agents.DDPG_v2 import DDPG_v2
import numpy as np
from agents.model_learner import Model_Learner
import os

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 500
MEMORY_CAPACITY = int(1e4)

#########################  training  ########################
RENDER = True
env = RealCircularPendulumEnv()
env.seed(1)

class argument_class(object):
    def __init__(self):
        self.env_name = 'RealHardware'
        self.num_episodes = int(1e3)
        self.seed = 1
        self.num_steps = 500
        self.gamma = 0.99
        self.tau = 0.01
        self.critic_lr = 5e-4
        self.actor_lr = 1e-4
        self.batch_size = 64
        self.replayBuffer_size = int(2e4)

args = argument_class()

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

########################################
######### the learning agent ###########
########################################
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
ddpg = DDPG_v2(sess, env, args)

path = 'C:/Users/dell/PycharmProjects/HardwarePendulum_Reduced/runfiles/data/DDPG2_RealCircularPendulum_22-08-2019_12-48-45'
ddpg.model_load(filepath=path)
n_eps = 100
n_steps = 800

for eps in range(n_eps):
    # s = env.reset(is_invert=True)
    s= env.reset()
    t_start = time.time()
    episode_reward = 0
    hardware_step_counter = 0
    for steps in range(n_steps):
        a = ddpg.choose_action(s)
        a = np.clip(a, -1, 1)

        s_, a,  r, done, info, hardware_step_counter, data_valid = env.step(a, hardware_step_counter)
        s = s_
        # env.render()
        if data_valid:
            episode_reward += r

        if (steps == n_steps-1):
            env.stop_pendulum()

            t_end = time.time()
            print("time:", int(t_end - t_start))
            print('Hardware step counter:', hardware_step_counter)
        # if done:
        #     break
    print('Episode {} ends with reward: {}'.format(eps, episode_reward))
    print('------------------------------')