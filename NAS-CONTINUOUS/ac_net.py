# -*- coding: utf-8 -*-
"""
The actor-critic network
"""
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np


# Used to initialize weights for policy and value output layers
# normalized_columns_initializer
def nc_initer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


# Entropy beta for loss function
ENTROPY_COEFFICIENT = 0.01
# Learning rates
POLICY_LR = 5e-4
VALUE_LR = 1e-2


class ACNet(object):
    def __init__(self, sess, state_len, actions_dim, action_bounds, scope):
        self.actions_dim = actions_dim
        self.state_len = state_len
        self.sess = sess
        self.scope = scope
        self.action_bounds = action_bounds
        self.layers_no = self.state_len//self.actions_dim
        self.trainer_p = tf.train.RMSPropOptimizer(learning_rate=POLICY_LR)
        self.trainer_v = tf.train.RMSPropOptimizer(learning_rate=VALUE_LR)
        # Build and initialize the AC net
        self.build_model()
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        with tf.variable_scope(self.scope):
            self.ac_input = tf.placeholder(
                tf.float32, shape=(None, self.layers_no, self.actions_dim), name='State')

            cell_sz = self.state_len**2

            # Critic sub-network
            with tf.variable_scope('V'):
                # RNN Cell
                rnn_cell = tf.contrib.rnn.GRUCell(cell_sz)
                outputs, state = tf.nn.dynamic_rnn(rnn_cell,
                                                   self.ac_input,
                                                   dtype=tf.float32)

                # Flatten the outputs
                flat = tf.reshape(
                    outputs, [-1, cell_sz], name='flatten_rnn_outputs')
                # Critic hidden network
                v_hid = layers.fully_connected(flat, cell_sz,
                                               weights_initializer=nc_initer(),
                                               activation_fn=tf.nn.relu6)
                # Value output
                self.value = \
                    layers.fully_connected(v_hid, 1,
                                           weights_initializer=nc_initer(),
                                           activation_fn=None)
            # Actor sub-network
            with tf.variable_scope('P'):
                # Actor hidden network
                # Actor's losses do not contribute to the RNN's updates
                p_hid = layers.fully_connected(tf.stop_gradient(flat), cell_sz*2,
                                               weights_initializer=nc_initer(),
                                               activation_fn=tf.nn.relu6)
                # Policy mean output
                self.policy_mean = \
                    layers.fully_connected(p_hid, self.actions_dim,
                                           weights_initializer=nc_initer(),
                                           activation_fn=tf.nn.tanh)

                # Scale it
                self.policy_mean = self.action_bounds[0] +\
                    self.policy_mean * \
                    (self.action_bounds[1]-self.action_bounds[0])

                # Policy sigma output
                self.policy_sigma = \
                    layers.fully_connected(p_hid, self.actions_dim,
                                           weights_initializer=nc_initer(),
                                           activation_fn=tf.nn.softmax)

                # Scale it
                self.policy_sigma *= (self.action_bounds[1] -
                                      self.action_bounds[0])/3
                # Clip sigma to [1e-4,1/3 of actions range]
                self.policy_sigma =\
                    tf.clip_by_value(
                        self.policy_sigma, 1e-4, (self.action_bounds[1]-self.action_bounds[0])/3)

                # Output the scaled actions
                self.actions_dist = tf.contrib.distributions.Normal(self.policy_mean, self.policy_sigma)

                # Sample an action from the distribution
                self.action = tf.to_int32(tf.squeeze(self.actions_dist.sample(1), axis=0))
                self.action = tf.clip_by_value(self.action, self.action_bounds[0], self.action_bounds[1])
            # Build the network's losses
            self.build_losses()

    def build_losses(self):
        with tf.variable_scope(self.scope):
            self.actions_t = tf.placeholder(shape=[None, self.layers_no, self.actions_dim], dtype=tf.float32,
                                            name='actions_tensor')

            self.target_v = tf.placeholder(shape=[None, self.layers_no, 1], dtype=tf.float32,
                                           name='target_v_tensor')

            # Critic losses
            with tf.variable_scope('V'):
                # Calculate advantage
                self.advantage = self.target_v - \
                    tf.reshape(
                        self.value, (-1, self.state_len//self.actions_dim, 1))
                # MSE
                self.value_loss = tf.reduce_mean(
                    tf.square(self.advantage), axis=0)
                # Get gradients from local network using local losses
                self.local_vars_v = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, self.scope+'/V')

                # Gradients calculation and application
                self.gradients_v = tf.gradients(
                    self.value_loss, self.local_vars_v)
                self.apply_gradients_v = self.trainer_v.apply_gradients(
                    zip(self.gradients_v, self.local_vars_v))

            # Actor losses
            with tf.variable_scope('P'):
                # Calculate H
                self.entropy = tf.reshape(self.actions_dist.entropy(
                ), (-1, self.layers_no, self.actions_dim))
                # Calculate log probabilities, given the current policy
                self.log_probs = self.actions_dist.log_prob(
                    tf.reshape(self.actions_t, (-1, self.actions_dim)))
                self.log_probs = tf.reshape(
                    self.log_probs, (-1, self.layers_no, self.actions_dim))
                # Calculate the policy loss
                self.policy_loss = self.log_probs * \
                    tf.reshape(tf.stop_gradient(self.advantage),
                               (-1, self.layers_no, 1))


                # Policy loss + entropy
                self.total_policy_loss = - \
                    tf.reduce_sum(self.policy_loss+ENTROPY_COEFFICIENT *
                                  self.entropy, axis=0)
                # Calculate the gradients with respect to the actors variables
                self.local_vars_p = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, self.scope+'/P')
                self.gradients_p = tf.gradients(
                    self.total_policy_loss, self.local_vars_p)
                # Define the application operation
                self.apply_gradients_p = self.trainer_p.apply_gradients(
                    zip(self.gradients_p, self.local_vars_p))
            # Add the grads to a dictionary in order to send them to
            # the parameter server

            self.grads = dict({'v': self.gradients_v, 'p': self.gradients_p})
            # Build the updater for the parameter server
            self.build_external_updater()

    # Get the initial weights in order to propagate them to workers
    def get_initial_weights(self):
        with tf.variable_scope(self.scope):
            local_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
            return self.sess.run([local_vars])

    def get_grads(self):
        return self.last_grads
    # Build the updater for the parameter server
    def build_external_updater(self):
        with tf.variable_scope(self.scope, reuse=True):
            # Operations to aply the actor, critic gradients
            apply_v_op = None
            apply_p_op = None
            # Critic
            with tf.variable_scope('V'):
                self.extern_grads_v = []
                # Apply all the gradient tensors
                for tensor in self.gradients_v:
                    self.extern_grads_v.append(
                        tf.placeholder(tf.float32, tensor.shape))
                    apply_v_op = self.trainer_v.apply_gradients(
                        zip(self.extern_grads_v, self.local_vars_v))
            # Actor
            with tf.variable_scope('P'):
                self.extern_grads_p = []
                # Apply all the tensors
                for tensor in self.gradients_p:
                    self.extern_grads_p.append(
                        tf.placeholder(tf.float32, tensor.shape))
                    apply_p_op = self.trainer_p.apply_gradients(
                        zip(self.extern_grads_p, self.local_vars_p))

            self.apply_grads_extern = [apply_p_op, apply_v_op]

    # Update the parameter server with the worker's gradients
    def update_with_grads(self, grads):
        with tf.variable_scope(self.scope, reuse=True):
            feed_dict = dict()
            v_grads = grads['v']
            p_grads = grads['p']
            for i in range(len(v_grads)):
                feed_dict[self.extern_grads_v[i]] = v_grads[i]
            for i in range(len(p_grads)):
                feed_dict[self.extern_grads_p[i]] = p_grads[i]
            self.sess.run([self.apply_grads_extern], feed_dict=feed_dict)

    def get_weights(self):
        with tf.variable_scope(self.scope, reuse=True):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def set_weights(self, new_weights):
        self._set_weights_func(new_weights)

    def _set_weights_func(self, new_weights):
        weights = self.get_weights()
        for i in range(len(weights)):
            self.sess.run(weights[i].assign(new_weights[i]))

    # Get predictions, given the current state
    def predict(self, state):
        action, policy_mean, policy_sigma, value = self.sess.run(
            [self.action, self.policy_mean, self.policy_sigma, self.value],
            feed_dict={self.ac_input: state})

        return action, policy_mean, policy_sigma, value

    # Train the network
    def fit(self, state, action, reward):

        # Reshape state, action, reward
        state = np.array(state)
        state = np.reshape(state, (-1, self.state_len //
                                   self.actions_dim, self.actions_dim))

        action = np.array(action)
        action = np.reshape(action, (-1, self.state_len //
                                     self.actions_dim, self.actions_dim))


        reward = np.array(reward)
        reward = np.reshape(reward, (-1, self.state_len//self.actions_dim, 1))

        feed_dict = {self.target_v: reward,
                     self.ac_input: state,
                     self.actions_t: action}

        v_l, p_l, e, grads, a_v, a_p = self.sess.run([self.value_loss,
                                                      self.policy_loss,
                                                      self.entropy,
                                                      self.grads,
                                                      self.apply_gradients_v,
                                                      self.apply_gradients_p],
                                                     feed_dict=feed_dict)

        # Store the gradients
        self.last_grads = grads
        return v_l, p_l, e, grads
