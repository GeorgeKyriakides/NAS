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
# Learning rate
LR = 1e-5


class ACNet(object):
    def __init__(self, sess, state_len, actions_no, scope):
        self.batch_size = 128
        self.actions_no = actions_no
        self.state_len = state_len
        self.sess = sess
        self.scope = scope
        self.trainer = tf.train.AdamOptimizer(learning_rate=LR)
        # Build and initialize the AC net
        self.build_model()
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        with tf.variable_scope(self.scope):
            self.ac_input = tf.placeholder(
                tf.float32, shape=(None, self.state_len), name='State')

            # Simple fully-connected network with policy and value outputs
            state_sq = (self.state_len)**2
            hid_1 = layers.fully_connected(self.ac_input, state_sq,
                                           activation_fn=tf.nn.softplus)

            # Policy output
            self.policy = \
                layers.fully_connected(hid_1, self.actions_no,

                                       activation_fn=tf.nn.softmax,
                                       weights_initializer=nc_initer(0.01),
                                       biases_initializer=None)
            # Value output
            self.value = \
                layers.fully_connected(hid_1, 1,
                                       activation_fn=None,
                                       weights_initializer=nc_initer(0.01),
                                       biases_initializer=None)
            self.build_losses()

    def build_losses(self):
        with tf.variable_scope(self.scope):
            self.actions_t = tf.placeholder(shape=[None, 1], dtype=tf.int32,
                                            name='actions_tensor')
            self.actions_onehot = tf.one_hot(self.actions_t, self.actions_no,
                                             dtype=tf.float32,
                                             name='actions_tensor_oh')
            self.target_v = tf.placeholder(shape=[None, 1], dtype=tf.float32,
                                           name='target_v_tensor')

            # Outputs responsible for rewards
            self.responsible_outputs = \
                tf.reduce_sum(self.policy * self.actions_onehot, [1])

            # Get the advantage
            self.advantage = self.target_v - tf.reshape(self.value, [-1])
            # MSE
            self.value_loss = 0.5 * \
                tf.reduce_mean(tf.square(self.target_v -
                                         tf.reshape(self.value, [-1])))
            # Get the distribution entropy
            self.entropy = - \
                tf.reduce_sum(self.policy * tf.log(1e-6+self.policy))
            # Get the policy loss
            self.policy_loss = - \
                tf.reduce_sum(tf.log(1e-6 + self.responsible_outputs)
                              )*tf.stop_gradient(self.advantage)

            # Get the total loss
            self.loss = 0.5 * self.value_loss + self.policy_loss - \
                self.entropy * ENTROPY_COEFFICIENT

            # Get gradients from local network using local losses
            local_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            # Clip by global norm
            self.grads, self.grad_norms = tf.clip_by_global_norm(
                self.gradients, 40)

            # Apply the gradients
            self.apply_gradients = self.trainer.apply_gradients(
                zip(self.grads, local_vars))
            # Build the updater for the parameter server
            self.build_external_updater()

    # Build the updater for the parameter server
    def build_external_updater(self):
        with tf.variable_scope(self.scope, reuse=True):
            self.extern_grads = []
            for tensor in self.grads:
                # Apply all the gradient tensors
                self.extern_grads.append(
                    tf.placeholder(tf.float32, tensor.shape))

            local_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
            self.apply_grads_extern = self.trainer.apply_gradients(
                zip(self.extern_grads, local_vars))

    # Update the parameter server with the worker's gradients
    def update_with_grads(self, grads):
        with tf.variable_scope(self.scope, reuse=True):
            feed_dict = dict()
            for i in range(len(grads)):
                feed_dict[self.extern_grads[i]] = grads[i]
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

        policy, value = self.sess.run(
            [self.policy, self.value],
            feed_dict={self.ac_input: state})
        return policy, value

    # Train the network
    def fit(self, state, action, reward):
        action = np.reshape(action, (len(action), 1))

        reward = np.reshape(reward, (len(reward), 1))

        feed_dict = {self.target_v: reward,
                     self.ac_input: state,
                     self.actions_t: action}
        v_l, p_l, e, g_n, v_n, grads, apply_grads = self.sess.run([self.value_loss,
                                                                   self.policy_loss,
                                                                   self.entropy,
                                                                   self.grad_norms,
                                                                   self.var_norms,
                                                                   self.grads,
                                                                   self.apply_gradients], feed_dict=feed_dict)

        return v_l, p_l, e, g_n, v_n, grads
