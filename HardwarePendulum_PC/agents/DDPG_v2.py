import time
import numpy as np
import tensorflow as tf
import argparse


class DDPG_v2:
    """docstring for DDPG"""

    def __init__(self, sess, env, args):
        self.s_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.sess = sess
        self.args = args

        self._build_graph()
        self.memory = Memory(self.args.replayBuffer_size, dims=2 * self.s_dim + self.act_dim + 1)
        self.saver = tf.train.Saver()

    def _build_graph(self):
        self._placehoders()
        self._actor_critic()
        self._loss_train_op()
        self.score = tf.Variable(0., trainable=False, dtype=tf.float32, name='score')
        self.score_summary = tf.summary.scalar('score', self.score)
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter('logs/')
        self.writer.add_graph(self.sess.graph)

    def _placehoders(self):
        with tf.name_scope('inputs'):
            self.current_state = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='s')
            self.reward = tf.placeholder(tf.float32, [None, 1], name='r')
            self.next_state = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='s_')
            self.is_training = tf.placeholder(tf.bool, name='is_training')

    def _actor_critic(self):
        self.actor, self.actor_summary = build_actor(self.current_state, self.act_dim, self.is_training)
        self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Actor')
        actor_ema = tf.train.ExponentialMovingAverage(decay=1 - self.args.tau)
        self.update_targetActor = actor_ema.apply(self.actor_vars)
        self.targetActor, _ = build_actor(self.next_state, self.act_dim, False,
                                          reuse=True, getter=get_getter(actor_ema))

        self.critic, self.critic_summary = build_critic(self.current_state, self.actor, self.act_dim)
        self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Critic')
        critic_ema = tf.train.ExponentialMovingAverage(decay=1 - self.args.tau)
        self.update_targetCritic = critic_ema.apply(self.critic_vars)
        self.targetCritic, _ = build_critic(self.next_state, self.targetActor, self.act_dim,
                                            reuse=True, getter=get_getter(critic_ema))

    def _loss_train_op(self):
        max_grad = 2
        with tf.variable_scope('target_q'):
            self.target_q = self.reward + self.args.gamma * self.targetCritic
        with tf.variable_scope('TD_error'):
            self.critic_loss = tf.squared_difference(self.target_q, self.critic)
        with tf.variable_scope('critic_grads'):
            self.critic_grads = tf.gradients(ys=self.critic_loss, xs=self.critic_vars)
            for ix, grad in enumerate(self.critic_grads):
                self.critic_grads[ix] = grad / self.args.batch_size
        with tf.variable_scope('C_train'):
            critic_optimizer = tf.train.AdamOptimizer(self.args.critic_lr, epsilon=1e-5)
            self.train_critic = critic_optimizer.apply_gradients(zip(self.critic_grads, self.critic_vars))
        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.critic, self.actor)[0]
        with tf.variable_scope('actor_grads'):
            self.actor_grads = tf.gradients(ys=self.actor, xs=self.actor_vars, grad_ys=self.a_grads)
            for ix, grad in enumerate(self.actor_grads):
                self.actor_grads[ix] = tf.clip_by_norm(grad / self.args.batch_size, max_grad)
        with tf.variable_scope('A_train'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                actor_optimizer = tf.train.AdamOptimizer(-self.args.actor_lr,
                                                         epsilon=1e-5)  # (- learning rate) for ascent policy
                self.train_actor = actor_optimizer.apply_gradients(zip(self.actor_grads, self.actor_vars))

    def choose_action(self, state):
        state = state[np.newaxis, :]  # single state
        return self.sess.run(self.actor, feed_dict={self.current_state: state,
                                                    self.is_training: False})[0]  # single action

    def train(self, episode=None, ep_reward=None):
        b_m = self.memory.sample(self.args.batch_size)
        b_s = b_m[:, :self.s_dim]
        b_a = b_m[:, self.s_dim: self.s_dim + self.act_dim]
        b_r = b_m[:, -self.s_dim - 1: -self.s_dim]
        b_s_ = b_m[:, -self.s_dim:]

        if episode is None:
            critic_feed_dict = {self.current_state: b_s, self.actor: b_a, self.reward: b_r, self.next_state: b_s_}
            self.sess.run([self.train_critic, self.update_targetCritic],
                          feed_dict=critic_feed_dict)
            actor_feed_dict = {self.current_state: b_s, self.next_state: b_s_, self.is_training: True}
            self.sess.run([self.train_actor, self.update_targetActor],
                          feed_dict=actor_feed_dict)
        else:
            update_score = self.score.assign(tf.convert_to_tensor(ep_reward, dtype=tf.float32))
            with tf.control_dependencies([update_score]):
                merged_score = tf.summary.merge([self.score_summary])
            critic_feed_dict = {self.current_state: b_s, self.actor: b_a, self.reward: b_r, self.next_state: b_s_}
            _, _, critic = self.sess.run([self.train_critic, self.update_targetCritic, self.critic_summary],
                                         feed_dict=critic_feed_dict)
            self.writer.add_summary(critic, episode)
            actor_feed_dict = {self.current_state: b_s, self.next_state: b_s_, self.is_training: True}
            merged = tf.summary.merge([merged_score, self.actor_summary])
            _, _, actor = self.sess.run([self.train_actor, self.update_targetActor, merged],
                                        feed_dict=actor_feed_dict)
            self.writer.add_summary(actor, episode)

    def perceive(self, state, action, reward, next_state, episode=None, ep_reward=None):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.memory.store_transition(state, action, reward, next_state)
        # Store transitions to replay start size then start training
        if self.memory.pointer > 10000:
            self.train(episode, ep_reward)
            # print("training!")

    # def perceive(self, state, action, reward, next_state, episode=None, ep_reward=None):
    #     # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
    #     self.memory.store_transition(state, action, reward, next_state)
    #     # Store transitions to replay start size then start training
    #     if self.memory.pointer > 20 * self.args.batch_size:
    #         self.train(episode, ep_reward)


    def model_save(self, filepath, step=None):
        save_path = self.saver.save(self.sess, save_path=filepath+'/saved_model', global_step=step)
        # print("Model saved in path: %s" % save_path)
        print("Model saved!")

    def model_load(self, filepath):
        # saver = tf.train.import_meta_graph('saved_models/' + filename + '.meta')
        # print(tf.train.latest_checkpoint('./saved_models/'))
        self.saver.restore(self.sess, tf.train.latest_checkpoint(filepath))
        print("Model successfully loaded!")



class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))

        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1
    def sample(self, n):
        # assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        # print("p:", self.pointer)
        # print("c:", self.capacity - 1)
        # print("min:", np.min((self.pointer, self.capacity)))
        indices = np.random.choice(np.min((self.pointer, self.capacity)), size=n)

        return self.data[indices, :]

    # def sample(self, n):
    #     # assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
    #     indices = np.random.choice(min(self.pointer, self.capacity), size=n)
    #     return self.data[indices, :]


def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    return ema_getter


def build_actor(s, act_dim, is_training, reuse=None, getter=None):
    hid1_size = 200
    hid2_size = 100
    with tf.variable_scope('Actor', reuse=reuse, custom_getter=getter):
        init_w = tf.random_normal_initializer(0., 0.3)
        init_b = tf.constant_initializer(0.1)
        hidden_1 = tf.layers.dense(s, hid1_size, activation=tf.nn.relu,
                                   kernel_initializer=init_w, bias_initializer=init_b, name='hidden_1')
        hid1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Actor/hidden_1')
        h1w_summary = tf.summary.histogram('h1w', hid1_vars[0])
        h1b_summary = tf.summary.histogram('h1b', hid1_vars[1])
        h1out_summary = tf.summary.histogram('h1out', hidden_1)
        h1_bn = tf.layers.batch_normalization(hidden_1, training=is_training, name='bn_1')
        h1bn_summary = tf.summary.histogram('h1bn', h1_bn)
        hidden_2 = tf.layers.dense(h1_bn, hid2_size, activation=tf.nn.relu,
                                   kernel_initializer=init_w, bias_initializer=init_b, name='hidden_2')
        hid2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Actor/hidden_2')
        h2w_summary = tf.summary.histogram('h2w', hid2_vars[0])
        h2b_summary = tf.summary.histogram('h2b', hid2_vars[1])
        h2out_summary = tf.summary.histogram('h2out', hidden_2)
        h2_bn = tf.layers.batch_normalization(hidden_2, training=is_training, name='bn_2')
        h2bn_summary = tf.summary.histogram('h2bn', h2_bn)
        actions = tf.layers.dense(h2_bn, act_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                  bias_initializer=init_b, name='action')
        action_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Actor/action')
        actw_summary = tf.summary.histogram('actw', action_vars[0])
        actb_summary = tf.summary.histogram('actb', action_vars[1])
        actout_summary = tf.summary.histogram('actout', actions)
    return actions, tf.summary.merge([h1w_summary, h1b_summary, h1out_summary, h1bn_summary,
                                      h2w_summary, h2b_summary, h2out_summary, h2bn_summary,
                                      actw_summary, actb_summary, actout_summary])


def build_critic(s, a, act_dim, reuse=None, getter=None):
    hid1_size = 200
    hid2_size = 100
    with tf.variable_scope('Critic', reuse=reuse, custom_getter=getter):
        init_w = tf.random_normal_initializer(0., 0.1)
        init_b = tf.constant_initializer(0.1)
        hidden_1 = tf.layers.dense(s, hid1_size, activation=tf.nn.relu,
                                   kernel_initializer=init_w, bias_initializer=init_b, name='hidden_1')
        hid1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Critic/hidden_1')
        h1w_summary = tf.summary.histogram('h1w', hid1_vars[0])
        h1b_summary = tf.summary.histogram('h1b', hid1_vars[1])
        h1out_summary = tf.summary.histogram('h1out', hidden_1)
        with tf.variable_scope('hidden_2'):
            w2_s = tf.get_variable('w2_s', [hid1_size, hid2_size], initializer=init_w)
            h2ws_summary = tf.summary.histogram('h2ws', w2_s)
            w2_a = tf.get_variable('w2_a', [act_dim, hid2_size], initializer=init_w)
            h2wa_summary = tf.summary.histogram('h2wa', w2_a)
            b2 = tf.get_variable('b2', [1, hid2_size], initializer=init_b)
            b2_summary = tf.summary.histogram('b2', b2)
            hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2_s) + tf.matmul(a, w2_a) + b2)
            h2out_summary = tf.summary.histogram('h2out', hidden_2)
        q = tf.layers.dense(hidden_2, 1, kernel_initializer=init_w, bias_initializer=init_b, name='q')
        q_summary = tf.summary.histogram('Q', q)
    return q, tf.summary.merge([h1w_summary, h1b_summary, h1out_summary,
                                h2ws_summary, h2wa_summary, b2_summary, h2out_summary,
                                q_summary])


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--env-name', default='HalfCheetah-v2',
                        help='environment to train on (default: HalfCheetah-v2)')
    parser.add_argument('--num-steps', type=int, default=1000,
                        help='number of environment steps (default: 1000)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-episodes', type=int, default=int(2e2),
                        help='number of frames to train (default: 1e3)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.01,
                        help='discount factor for rewards (default: 0.01)')
    parser.add_argument('--critic-lr', type=float, default=5e-4,
                        help='critic learning rate (default: 2e-4)')
    parser.add_argument('--actor-lr', type=float, default=1e-4,
                        help='actor learning rate (default: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='number of batch size to train (default: 64)')
    parser.add_argument('--replayBuffer-size', type=int, default=int(2e2),
                        help='size of the replay buffer (default: 2e4)')
    parser.add_argument('--log-dir', default='logs/',
                        help='directory to save agent logs (default: "logs/")')
    args = parser.parse_args()
    return args

def learn(args, env, agent):
    render = False
    var = 3  # control exploration
    start = time.time()
    for e in range(args.num_episodes):
        obs = env.reset()
        ep_reward = 0
        if var >= 0.1:
            var *= .995  # decay the action randomness
        for j in range(args.num_steps):
            if render:
                env.render()
            action = agent.choose_action(obs)
            # Add exploration noise
            action = np.clip(np.random.normal(action, var), -2, 2)  # add randomness to action selection for exploration
            next_obs, reward, done, info = env.step(action)
            ep_reward += reward
            if j == args.num_steps - 1:
                agent.perceive(obs, action, reward * 0.1, next_obs, episode=e, ep_reward=ep_reward)
                end = time.time()
                total_num_steps = (e + 1) * args.num_steps
                print('Episode:', e, 'FPS:', int(total_num_steps / (end - start)),
                      'Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                if ep_reward > 2000:
                    render = True
                break
            else:
                agent.perceive(obs, action, reward * 0.1, next_obs)
            obs = next_obs

    agent.sess.close()