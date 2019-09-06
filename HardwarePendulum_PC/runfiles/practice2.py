import sys
sys.path.append('/Users/indurance/PycharmProjects/MasterProject')
import gym
import tensorflow as tf
import time
from envs.real_circular_pendulum import RealCircularPendulumEnv2
from agents.DDPG_v2 import DDPG_v2
import numpy as np
from agents.model_learner import Model_Learner
import os
import utilits
from agents.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise


env = RealCircularPendulumEnv2()

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
a_max = env.action_space.high
a_min = env.action_space.low
print('State space: Dimension {} Bounds {} - {}'.format(env.observation_space.shape, env.observation_space.low,
                                                       env.observation_space.high))
print('Action space: {0} - {1}'.format(a_min, a_max))

var = 1  # control exploration
noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(a_dim), sigma=float(var) * np.ones(a_dim), theta=0.5)

def pid(s):
    position = 5*(s[0] - 10000) + 100*s[2]
    position = 0
    balance = -40*(s[1] - 3085) - 50*s[3]
    print("s:", s)
    print("pwm:", balance - position)
    return (balance - position)/3000

# a = pid([0.15707963, 3.12626781, 0, 0])
# print(a)
# input('pause')
MAX_EPISODE = 2
MAX_STEP = 1000

for ep in range(MAX_EPISODE):
    s = env.reset(is_invert=True)
    print("-----------------------------")
    print("Episode {} starts".format(ep))
    t_start = time.time()
    hardware_step_counter = 0
    for step in range(MAX_STEP):
        # a = -0.4
        # a = np.random.uniform(low=0.4, high=0.5)
        # a = np.random.uniform(low=-0.4, high=0.5)
        # time.sleep(0.002)
        # t1 = int(round(time.time() * 1000))
        # a = noise()
        a = pid(s)
        print(a)
        print("s", s)
        s_, a,  r, done, info, hardware_step_counter, data_valid = env.step(a, hardware_step_counter)
        # t2 = int(round(time.time() * 1000))
        # print("t: ", t2-t1)
        # print("state: ", s, "action", a, "next_state", s_, "t: ", int(t2-t1))
        s = s_
        # print("step:", step)
        if (step == MAX_STEP-1):
            env.stop_pendulum()
            t_end = time.time()
            print("time:", int(t_end - t_start))
            print('Hardware step counter:', hardware_step_counter)


print('END...')
