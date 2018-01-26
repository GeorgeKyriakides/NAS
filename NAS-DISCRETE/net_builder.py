# -*- coding: utf-8 -*-
"""
Network architecture builder
"""
import tensorflow as tf
import tensorflow.contrib.layers as layerz
import numpy as np


class Model(object):
    def __init__(self, input_t, output_t, target_t, loss, loss_op, train_op,
                 accuracy, dropouts):

        self.input_t = input_t
        self.output_t = output_t
        self.target_t = target_t
        self.loss = loss
        self.loss_op = loss_op
        self.train_op = train_op
        self.accuracy = accuracy
        self.dropouts = dropouts

    # Set the dropout layers to training/predction mode
    def set_training(self, mode=True):
        for dropout in self.dropouts:
            dropout.is_training = mode


class NetBuilder(object):

    def __init__(self, input_shape, trainable=True, actions=None):
        self.trainable = trainable
        # self.initer=TruncatedNormal(mean=0.0,stddev=0.0015,seed=1354567)

        self.input_shape = input_shape
        if actions is None:
            conv_filters = []  # [16,32,64]
            conv_kernels = []  # [1,2,3]
            conv_list = []
            for f in conv_filters:
                for k in conv_kernels:
                    conv_list.append(str(f)+','+str(k))
            self.types = {
                'fc': [64, 128, 256, 512, 1024], 'dout': [0.1, 0.3, 0.5], 'conv2d': conv_list
            }
            self.actions = ['NONE-NONE']
            for key in self.types.keys():
                for parameter in self.types[key]:
                    self.actions.append(str(key)+'-'+str(parameter))
        else:
            self.actions = actions
        print(self.actions)
        self.exp_no = 0

    def build_net(self, state):
        # Set the experiment number (scope)
        self.exp_no += 1
        state = state.copy()
        # Discard the starting state
        state = state[1:]
        state_len = len(state)
        # Break the state into layers
        layers = [state[x:x+self.actions_no()-1]
                  for x in range(0, state_len, self.actions_no()-1)]
        dropouts = []
        # Define the network's input, output, target tensors
        input_t = None
        output_t = None
        target_t = tf.placeholder(tf.int32, [None, 10], name="targets")
        input_t = tf.placeholder(
            tf.float32, shape=self.input_shape,
            name='Input'+str(self.exp_no))
        # The first previous layer is the input tensor
        prev_layer = input_t
        # Create all the hidden layers
        for layer in layers:
            layer, params = self.get_layer_action(layer)
            prev_layer, dropouts = self.create_layer(layer, params,
                                                     prev_layer, dropouts)
        # Conect the last to the classification layer
        output_t = layerz.fully_connected(prev_layer, 10,
                                          activation_fn=tf.nn.softplus)
        # Simple cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=output_t, labels=target_t, name="loss")
        # Define the training/optimization operations
        loss_op = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss_op)
        correct = tf.equal(tf.argmax(output_t, 1),
                           tf.argmax(target_t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        # Create the model and return it
        return Model(input_t, output_t, target_t, loss, loss_op, train_op,
                     accuracy, dropouts)

    # Get the layer from the action
    def get_layer_action(self, layer):
        # If no action was taken return none
        if np.sum(layer) == 0:
            return 'NONE', 'NONE'

        action_index = np.nonzero(layer)[0][0] + 1

        action = self.actions[action_index]
        layer_type = action.split('-')[0]
        parameter = action.split('-')[1]
        return layer_type, parameter

    def create_layer(self, layer_type, parameter, input_t, dropouts):
        next_t = input_t
        if layer_type == 'fc':
            next_t = layerz.fully_connected(input_t, int(parameter),
                                            activation_fn=tf.nn.relu,
                                            trainable=self.trainable)

        elif layer_type == 'dout':
            next_t = layerz.dropout(input_t, 1-float(parameter))
            dropouts.append(next_t)

        return next_t, dropouts

    def actions_no(self):
        return len(self.actions)
