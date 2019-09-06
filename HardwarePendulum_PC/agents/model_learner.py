import tensorflow as tf
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt

# np.random.seed(1)
# tf.set_random_seed(2)

class Model_Learner(object):
    def __init__(
            self,
            env,
            num_actions,
            num_features,
            rollout_agent,
            learning_rate=1e-2,
            reward_decay=0.95,
            output_graph=True,
            num_hidden_layers=1,
            num_hidden_units_d=300,
            num_hidden_layers_d=1,
            num_hidden_units_r=100,
            num_hidden_layers_r=2,
            dataset_size=int(4 * 1e3),
    ):
        self.env = env
        self.n_actions = num_actions
        self.n_features = num_features
        self.lr = learning_rate
        self.decay = reward_decay
        self.output_graph = output_graph
        self.eps_states, self.eps_actions, self.eps_rewards = [], [], []
        self.eps_advantage_rewards = []
        self.rollout_agent = rollout_agent

        self.num_hidden_units_d = num_hidden_units_d
        self.num_hidden_layers_d = num_hidden_layers_d
        self.num_hidden_units_r = num_hidden_units_r
        self.num_hidden_layers_r = num_hidden_layers_r

        self.n_hidden_layers = num_hidden_layers
        self.activation = tf.nn.relu
        self.experience_dataset = Dataset(size=dataset_size)
        self.rollouts_length = 1000
        self.rollouts_collection(self.env, self.experience_dataset, num_rollouts=1, len_rollouts=self.rollouts_length)

        self.states, self.actions, self.next_states, self.next_rewards = self.place_holder_setup()
        self.loss, self.optimizer, self.model_output = self.model_dynamics_setup()
        self.r_loss, self.r_optimizer, self.r_model_output = self.model_reward_setup()

        self.sess = tf.Session()
        if output_graph:
            self.writer = tf.summary.FileWriter('logs/', self.sess.graph)
        self.initializer = tf.global_variables_initializer()
        self.sess.run(self.initializer)

    def rollouts_collection(self, env, data_set, len_rollouts=500, num_rollouts=20):
        for _ in range(num_rollouts):
            state = env.reset()
            data_valid = False
            counter = 0
            for t in range(len_rollouts):
                # action = env.action_space.sample()
                # action = np.random.uniform(env.action_space.low, env.action_space.high)
                action = self.rollout_agent.choose_action(state)
                next_state, action, reward, done, _, counter, data_valid = env.step(action, counter)
                if data_valid:
                    data_set.add_data_pair(state, action, next_state, reward, done)
                state = next_state

                if (t == len_rollouts - 1):
                    env.stop_pendulum()

        print('Rollouts collection completed ...')
        print('Experience Dataset with %d rollouts' % (data_set.length()))
        print('Mean statistics: ')
        print(self.experience_dataset.state_mean)
        print(self.experience_dataset.action_mean)
        print(self.experience_dataset.delta_state_mean)
        print('Std statistics: ')
        print(self.experience_dataset.state_std)
        print(self.experience_dataset.action_std)
        print(self.experience_dataset.delta_state_std)

    def place_holder_setup(self):
        states = tf.placeholder(tf.float32, shape=[None, self.n_features], name='states')
        actions = tf.placeholder(tf.float32, shape=[None, self.n_actions], name="actions_num")
        next_states = tf.placeholder(tf.float32, shape=[None, self.n_features], name='next_states')
        next_rewards = tf.placeholder(tf.float32, shape=[None, 1], name='next_states')
        return states, actions, next_states, next_rewards

    def model_dynamics_setup(self):
        #### declare normalized tensors ####
        states_n = normalize(self.states, self.experience_dataset.state_mean,
                                  self.experience_dataset.state_std)
        actions_n = normalize(self.actions, self.experience_dataset.action_mean,
                                  self.experience_dataset.action_std)
        print(states_n)
        print(actions_n)
        states_actions_n = tf.concat([states_n, actions_n], 1)

        #### define Net architecture ####
        num_hidden_units = self.num_hidden_units_d
        num_hidden_layer = self.num_hidden_layers_d
        # regularizer
        # regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        regularizer = None

        # input layer
        layer = tf.layers.dense(
            states_actions_n,
            num_hidden_units,
            activation=self.activation,
            kernel_regularizer=regularizer
        )
        # hidden layer
        for l in range(num_hidden_layer):
            layer = tf.layers.dense(
                layer,
                num_hidden_units,
                activation=self.activation,
            )
        # output layer
        layer = tf.layers.dense(
            layer,
            self.n_features,
            activation=None,
        )

        print(layer)
        delta_states = unnormalize(layer,  self.experience_dataset.delta_state_mean, self.experience_dataset.delta_state_std)
        next_states_pred = delta_states + self.states
        delta_states_real = self.next_states - self.states
        delta_states_real_n = normalize(delta_states_real,  self.experience_dataset.delta_state_mean, self.experience_dataset.delta_state_std)
        loss = tf.reduce_mean((delta_states_real_n - layer)**2)
        l2_loss = tf.losses.get_regularization_loss()
        loss += l2_loss
        # loss = tf.reduce_mean((next_states_pred - self.next_states) ** 2)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        return loss, optimizer, next_states_pred

    def model_reward_setup(self):
        #### declare normalized tensors ####
        states_n = normalize(self.states, self.experience_dataset.state_mean,
                                  self.experience_dataset.state_std)
        actions_n = normalize(self.actions, self.experience_dataset.action_mean,
                                  self.experience_dataset.action_std)
        print(states_n)
        print(actions_n)
        states_actions_n = tf.concat([states_n, actions_n], 1)

        #### define Net architecture ####
        num_hidden_units = self.num_hidden_units_r
        num_hidden_layer = self.num_hidden_layers_r
        # regularizer
        # regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        regularizer = None

        # input layer
        layer = tf.layers.dense(
            states_actions_n,
            num_hidden_units,
            activation=self.activation,
            kernel_regularizer=regularizer
        )
        # hidden layer
        for l in range(num_hidden_layer):
            layer = tf.layers.dense(
                layer,
                num_hidden_units,
                activation=self.activation,
            )
        # output layer
        layer = tf.layers.dense(
            layer,
            1,
            activation=None,
        )

        print("Reward model output:{}".format(layer))
        next_reward_pred = layer
        loss = tf.reduce_mean((self.next_rewards - layer)**2)
        l2_loss = tf.losses.get_regularization_loss()
        loss += l2_loss
        # loss = tf.reduce_mean((next_states_pred - self.next_states) ** 2)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        return loss, optimizer, next_reward_pred

    def model_training(self, training_epochs=40, training_batch_size=512):
        print('Training Dyanamics Model  ...')

        for i in range(training_epochs):
            for state, action, next_state, reward, done in self.experience_dataset.random_iterator(training_batch_size):
                _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.states: state,
                                                                     self.actions: action,
                                                                     self.next_states: next_state})
            print('dynamic loss: %f' % (loss))

        print('Dynamics Training completed!')
        print('Final dynamic loss: %f' % (loss))

    def reward_model_training(self, training_epochs=40, training_batch_size=512):
        print('Training Reward Model  ...')

        for i in range(training_epochs):
            for state, action, next_state, reward, done in self.experience_dataset.random_iterator(training_batch_size):
                _, loss = self.sess.run([self.r_optimizer, self.r_loss], feed_dict={self.states: state,
                                                                     self.actions: action,
                                                                     self.next_rewards: reward})
            print('reward loss: %f' % (loss))

        print('Reward Model Training completed!')
        print('Final reward loss: %f' % (loss))

    def simulated_step(self, state, action):
        next_state_pred, next_reward_pred = self.sess.run([self.model_output, self.r_model_output],
                                        feed_dict={self.states: np.array(state).reshape(1, self.n_features),
                                                   self.actions: np.array(action).reshape(1, self.n_actions)})
        return np.array(np.squeeze(next_state_pred)), next_reward_pred

    def model_testing(self):
        testing_length = self.rollouts_length
        states_pred_array = []
        state_pred = self.experience_dataset._states[0]
        action_array = self.experience_dataset._actions

        for i in range(len(self.experience_dataset._actions)-1):
            states_pred_array.append(np.ravel(state_pred))
            next_state_pred = self.sess.run(self.model_output, feed_dict={self.states: np.array(state_pred).reshape(1, self.n_features),
                                                                          self.actions: action_array[i].reshape(1, self.n_actions)})
            state_pred = next_state_pred

        # print(states_pred_array)
        # print(self.experience_dataset._states)

        states_real = np.array(self.experience_dataset._states[0:testing_length])
        states_pred = np.array(states_pred_array[0:testing_length])

        #axes = plt.subplot(2, 2)
        rows = int(np.sqrt(self.n_features))
        cols = self.n_features // rows
        f, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        f.suptitle('Model predictions (red) versus ground truth (black) with training data')
        for j, ax in enumerate(axes.ravel()):
            ax.set_title('state {0}'.format(j))
            ax.plot(states_real[:, j], color='k')
            ax.plot(states_pred[:, j], color='r')
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()

    def model_testing_testdata(self):
        test_dataset = Dataset()
        testing_length = 500
        self.rollouts_collection(self.env, test_dataset, len_rollouts=testing_length, num_rollouts=1)
        states_pred_array = []
        state_pred = test_dataset._states[0]
        action_array = test_dataset._actions

        for i in range(len(test_dataset._actions)-1):
            states_pred_array.append(np.ravel(state_pred))
            next_state_pred = self.sess.run(self.model_output, feed_dict={self.states: np.array(state_pred).reshape(1, self.n_features),
                                                                          self.actions: action_array[i].reshape(1, self.n_actions)})
            state_pred = next_state_pred

        # print(states_pred_array)
        # print(test_dataset._states)

        states_real = np.array(test_dataset._states[0:testing_length])
        states_pred = np.array(states_pred_array[0:testing_length])

        #axes = plt.subplot(2, 2)
        rows = int(np.sqrt(self.n_features))
        cols = self.n_features // rows
        f, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        f.suptitle('Model predictions (red) versus ground truth (black) with testing data')
        for j, ax in enumerate(axes.ravel()):
            ax.set_title('state {0}'.format(j))
            ax.plot(states_real[:, j], color='k')
            ax.plot(states_pred[:, j], color='r')
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()


    def entire_model_testing(self):
        testing_length = self.rollouts_length
        states_pred_array = []
        reward_pred_array = []
        state_pred = self.experience_dataset._states[0]
        action_array = self.experience_dataset._actions

        for i in range(len(self.experience_dataset._actions)-1):
            states_pred_array.append(np.ravel(state_pred))
            next_state_pred, reward = self.sess.run([self.model_output, self.r_model_output], feed_dict={self.states: np.array(state_pred).reshape(1, self.n_features),
                                                                          self.actions: action_array[i].reshape(1, self.n_actions)})
            state_pred = next_state_pred
            reward_pred_array.append(np.ravel(reward))

        # print(states_pred_array)
        # print(self.experience_dataset._states)

        states_real = np.array(self.experience_dataset._states[0:testing_length])
        states_pred = np.array(states_pred_array[0:testing_length])
        rewards_real = np.array(self.experience_dataset._rewards[0:testing_length])
        rewards_pred = np.array(reward_pred_array[0:testing_length])
        states_real = np.hstack((states_real, rewards_real))
        states_pred = np.hstack((states_pred, rewards_pred))

        # print('rewards_real:', rewards_real.ravel())
        # print('rewards_pred:', rewards_pred.ravel())

        rows = int(np.sqrt(self.n_features + 1))
        cols = int(np.ceil((self.n_features + 1) / rows))
        f, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        f.suptitle('Model predictions (red) versus ground truth (black) with training data')
        for j, ax in enumerate(axes.ravel()[:self.n_features]):
            ax.set_title('state {0}'.format(j))
            ax.plot(states_real[:, j], color='k')
            ax.plot(states_pred[:, j], color='r')
        axes.ravel()[self.n_features].set_title('Reward')
        axes.ravel()[self.n_features].plot(states_real[:, -1], color='k')
        axes.ravel()[self.n_features].plot(states_pred[:, -1], color='r')
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()

    def entire_model_testing_testdata(self):
        test_dataset = Dataset()
        testing_length = 200
        self.rollouts_collection(self.env, test_dataset, len_rollouts=testing_length, num_rollouts=1)
        states_pred_array = []
        reward_pred_array = []
        state_pred = test_dataset._states[0]
        action_array = test_dataset._actions

        for i in range(len(test_dataset._actions)-1):
            states_pred_array.append(np.ravel(state_pred))
            next_state_pred, reward = self.sess.run([self.model_output, self.r_model_output], feed_dict={self.states: np.array(state_pred).reshape(1, self.n_features),
                                                                          self.actions: action_array[i].reshape(1, self.n_actions)})
            state_pred = next_state_pred
            reward_pred_array.append(np.ravel(reward))

        # print(states_pred_array)
        # print(self.experience_dataset._states)

        states_real = np.array(test_dataset._states[0:testing_length])
        states_pred = np.array(states_pred_array[0:testing_length])
        rewards_real = np.array(test_dataset._rewards[0:testing_length])
        rewards_pred = np.array(reward_pred_array[0:testing_length])
        states_real = np.hstack((states_real, rewards_real))
        states_pred = np.hstack((states_pred, rewards_pred))

        # print('rewards_real:', rewards_real.ravel())
        # print('rewards_pred:', rewards_pred.ravel())

        rows = int(np.sqrt(self.n_features + 1))
        cols = int(np.ceil((self.n_features + 1) / rows))
        f, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        f.suptitle('Model predictions (red) versus ground truth (black) with testing data')
        for j, ax in enumerate(axes.ravel()[:self.n_features]):
            ax.set_title('state {0}'.format(j))
            ax.plot(states_real[:, j], color='k')
            ax.plot(states_pred[:, j], color='r')
        axes.ravel()[self.n_features].set_title('Reward')
        axes.ravel()[self.n_features].plot(states_real[:, -1], color='k')
        axes.ravel()[self.n_features].plot(states_pred[:, -1], color='r')
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()


def normalize(x, mean, std, eps=1e-8):
    return (x - mean) / (std + eps)

def unnormalize(x, mean, std):
    return x * std + mean

class Dataset(object):
    def __init__(self, size=int(4 * 1e3)):
        self._size = size
        self._pointer = 0
        self._states = []
        self._actions = []
        self._state_action_pairs = []
        self._next_states = []
        self._rewards = []
        self._dones = []

    def add_data_pair(self, state, action, next_state, reward, done):
        if self._pointer < self._size:
            self._states.append(np.ravel(state))
            self._actions.append(np.ravel(action))
            self._state_action_pairs.append(np.append(np.ravel(state), np.ravel(action)))
            self._next_states.append(np.ravel(next_state))
            self._rewards.append(np.ravel(reward))
            self._dones.append(done)
        else:
            index = self._pointer % self._size
            self._states[index] = np.ravel(state)
            self._actions[index] = np.ravel(action)
            self._state_action_pairs[index] = np.append(np.ravel(state), np.ravel(action))
            self._next_states[index] = np.ravel(next_state)
            self._rewards[index] = np.ravel(reward)
            self._dones[index] = done
        self._pointer += 1

    # def add_data_pair(self, state, action, next_state, reward, done):
    #     self._states.append(np.ravel(state))
    #     self._actions.append(np.ravel(action))
    #     self._state_action_pairs.append(np.append(np.ravel(state), np.ravel(action)))
    #     self._next_states.append(np.ravel(next_state))
    #     self._rewards.append(np.ravel(reward))
    #     self._dones.append(done)

    def add_data_set(self, new_dataset):
        self._states += new_dataset._states
        self._actions += new_dataset._actions
        self._state_action_pairs += new_dataset._state_action_pairs
        self._next_states += new_dataset._next_states
        self._rewards += new_dataset._rewards
        self._dones += new_dataset._dones

    def random_iterator(self, batch_size):
        """
        Iterate once through all (s, a, r, s') in batches in a random order
        """
        all_indices = np.nonzero(np.logical_not(self._dones))[0]
        np.random.shuffle(all_indices)

        states = np.asarray(self._states)
        actions = np.asarray(self._actions)
        next_states = np.asarray(self._next_states)
        rewards = np.asarray(self._rewards)
        dones = np.asarray(self._dones)

        i = 0
        while i < len(all_indices):
            indices = all_indices[i:i+batch_size]

            yield states[indices], actions[indices], next_states[indices], rewards[indices], dones[indices]

            i += batch_size

    def random_state_generate(self):
        index = np.random.choice(self.length(), 1)
        # print('random index: {}'.format(index))
        return self._states[int(index)]

    ##### Statistics #####
    @property
    def is_empty(self):
        return len(self._actions) == 0

    def length(self):
        return len(self._actions)

    @property
    def state_mean(self):
        return np.mean(self._states, axis=0)

    @property
    def state_std(self):
        return np.std(self._states, axis=0)

    @property
    def action_mean(self):
        return np.mean(self._actions, axis=0)

    @property
    def action_std(self):
        return np.std(self._actions, axis=0)

    @property
    def delta_state_mean(self):
        return np.mean(np.array(self._next_states) - np.array(self._states), axis=0)

    @property
    def delta_state_std(self):
        return np.std(np.array(self._next_states) - np.array(self._states), axis=0)