import sys
sys.path.append('/Users/indurance/PycharmProjects/MasterProject')
import gym
import tensorflow as tf
import time
from envs.real_circular_pendulum import RealCircularPendulumEnv
from agents.DDPG_v2 import DDPG_v2
import numpy as np
from agents.model_learner import Model_Learner
import os
import utilits
from agents.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

#####################  hyper parameters  ####################
class argument_class(object):
    def __init__(self):
        self.env_name = 'RealCircularPendulum'
        self.num_episodes = int(400)
        self.seed = 1
        self.num_steps = 800
        self.gamma = 0.99
        self.tau = 0.01
        self.critic_lr = 5e-4    # 1e-3
        self.actor_lr = 1e-4
        self.batch_size = 128
        self.replayBuffer_size = int(1e5)


args = argument_class()

env = RealCircularPendulumEnv()

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
a_max = env.action_space.high
a_min = env.action_space.low
print('State space: Dimension {} Bounds {} - {}'.format(env.observation_space.shape, env.observation_space.low,
                                                       env.observation_space.high))
print('Action space: {0} - {1}'.format(a_min, a_max))
# input('pause')

##################### data directory setup ###############
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
exp_name = '{0}_{1}_{2}'.format('DDPG2',
                                'RealCircularPendulum',
                                time.strftime("%d-%m-%Y_%H-%M-%S"))
exp_dir = os.path.join(data_dir, exp_name)
assert not os.path.exists(exp_dir), \
    'Experiment directory {0} already exists. Either delete the directory, or run the experiment with a different name'.format(
        exp_dir)
os.makedirs(exp_dir, exist_ok=True)
# print(exp_dir)
# wait = input("PRESS ENTER TO CONTINUE.")

########################################
######### the learning agent ###########
########################################

sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
ddpg = DDPG_v2(sess, env, args)

########## load model ###########
# path = 'C:/Users/dell/PycharmProjects/HardwarePendulum/runfiles/data/DDPG2_CircularPendulum_31-07-2019_costs += .5 * (theta_dot/10)**2 + .05 * (alpha_dot/20)**212-54-33'
# path = 'C:/Users/dell/PycharmProjects/HardwarePendulum/runfiles/data/DDPG2_RealCircularPendulum_14-08-2019_12-01-21'
# path = 'C:/Users/dell/PycharmProjects/HardwarePendulum_Reduced/runfiles/data/DDPG2_RealCircularPendulum_21-08-2019_17-51-35' nice past data
####
# path = 'C:/Users/dell/PycharmProjects/HardwarePendulum_Reduced/runfiles/data/DDPG2_RealCircularPendulum_02-09-2019_12-54-13'
# ddpg.model_load(filepath=path)
# data = np.load(path + "/experience_dataset.npy")
# ddpg.memory.data[:data.shape[0], :] = data
# print(data.shape[0])
# ddpg.memory.pointer = data.shape[0]
#####
#################################

########## exploration noise ###########
#--- OrnsteinUhlenbeck Noise ----#
# var = 0.1  # control exploration
# noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(a_dim), sigma=float(var) * np.ones(a_dim), theta=1)
#--- Normal Noise ----#
# var = 0.3
var = 0.2
noise = NormalActionNoise(mu=np.zeros(a_dim), sigma=float(var) * np.eye(a_dim))


def policy_evaluation(env, agent, eval_episode_length, initial_state=None):
    """Mind the action space restriction"""
    if initial_state != None:
        s = env.reset(initial_state)
    else:
        s = env.reset()
    noise.reset(sigma=0.01)
    epi_reward = 0
    counter = 0
    data_valid = False

    for j in range(eval_episode_length):
        a = agent.choose_action(s) + noise()  # no exploration noise
        a = np.clip(a, -1, 1)
        s_, a,  r, done, info, counter, data_valid = env.step(a, counter)

        if data_valid:
            epi_reward += r
            s = s_

        if (j == eval_episode_length-1):
            env.stop_pendulum()

    return epi_reward

t1 = time.time()
training_flag = True
iteration_counter = 0
reward_list = []
RENDER = False

N_plan = 1
for i in range(args.num_episodes):
    s = env.reset()
    start = time.time()
    s_eval = s
    ep_reward = .0
    ep_reward_eval = .0
    # plan_flag = bool(i % N_plan != 0)  # if use the learned model to plan
    plan_flag = False
    ep_type = ['Ture Environment', 'Learned Model'][int(plan_flag)]
    noise.reset(sigma=var)
    hardware_step_counter = 0
    p_eval_reward = 0
    if var >= 0.01:
        var *= .998  # decay the action randomness
        # var = 0
        pass

    for j in range(args.num_steps):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s) + noise()
        # a = -ddpg.choose_action(s)
        a = np.clip(a, -1, 1)  # add randomness to action selection for exploration (Remember to recover)

        if plan_flag == False:
            s_, a,  r, done, info, hardware_step_counter, data_valid = env.step(a, hardware_step_counter)
            # print("re: ", 0.1 * r)
            # print("state: ", s, "action", a, "next_state", s_)
        else:
            s_, r = learned_model.simulated_step(s, a)
            s_ = s_.ravel()
            r = float(r)
            ep_reward_eval += r
            # print('next state: {}'.format(s_.shape))
            # print('next state: {}'.format(r.shape))

        if data_valid:
            ep_reward += r

        if j == args.num_steps - 1:
            end = time.time()
            env.stop_pendulum()
            print('---------------------------------')
            print('######### Episode Info ##########')
            print('Episode:', i, 'Steps:', j, 'Time:', int(end-start), ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )

            if data_valid:
                ddpg.perceive(s, a, r * 0.1, s_, episode=i, ep_reward=ep_reward)

        else:
            if data_valid:
                ddpg.perceive(s, a, r * 0.1, s_)

        if data_valid:
            s = s_

    # training_flag = bool(np.absolute((ep_reward - ep_reward_eval) / ep_reward) > 0.2) and ep_reward < -600
    training_flag = False

    if plan_flag == False:
        iteration_counter += 1
        # print('Simulated Reward: {}'.format(ep_reward_eval))
        # print('Model Uncertainty: {}'.format(np.absolute((ep_reward - ep_reward_eval) / ep_reward)))
        # p_eval_reward = policy_evaluation(env, ddpg, args.num_steps)
        # print('Policy Evaluation Reward: {}'.format(int(p_eval_reward)))
        reward_list.append([iteration_counter, int(ep_reward)])
        np.savetxt(exp_dir + '/epsiode_data', reward_list, fmt='%4d')
        utilits.save_single_reward_curve1(np.array(reward_list)[:, 1], exp_dir)
        np.save(exp_dir + '/experience_dataset', ddpg.memory.data)

    if i % 10 == 0 and i != 0:
        print('---------------------------------')
        print('Evaluating......')
        p_eval_reward = policy_evaluation(env, ddpg, args.num_steps)
        print("Evaluated Episode Reward: ", p_eval_reward)
        time.sleep(20)

    if training_flag:
        learned_model.model_training(training_epochs=10)
        learned_model.reward_model_training(training_epochs=5)

    ddpg.model_save(filepath=exp_dir)
    print('######### Episode End ###########')
    env.stop_pendulum()

print('Running time: ', time.time() - t1)