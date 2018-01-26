# -*- coding: utf-8 -*-
"""
Network architecture builder
"""
import tensorflow as tf
import tensorflow.contrib.layers as layerz
import numpy as np


# Class to store the model
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

    def __init__(self, input_shape, actions_no, trainable=True):
        # If trainable is False, only the last layer
        # of the constructed network is trainable
        self.trainable = trainable
        self.actions_no = actions_no
        self.input_shape = input_shape
        self.exp_no = 0


    def build_net(self, state):
        # Set the experiment number (scope)
        self.exp_no += 1
        self.current_layer = 1
        state = state.copy()
        # Discard the starting state
        state = state[self.actions_no:]
        state_len = len(state)
        # Break the state into layers
        layers = [state[x:x+self.actions_no]
                  for x in range(0, state_len, self.actions_no)]
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
            prev_layer = self.create_layer(layer, prev_layer)
            dropouts.append(prev_layer)

        # Flatten the outputs and conect them to the classification layer
        flatout = tf.contrib.layers.flatten(prev_layer)
        output_t = layerz.fully_connected(flatout, 10,
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

    def create_layer(self, layer, prev_layer):

        # Layer's name
        c_l = str(self.exp_no)+'_'+str(self.current_layer)
        # Weights initializer
        x_i = tf.contrib.layers.xavier_initializer

        # Get the layer parameters
        filter_h = int(layer[0])
        filter_w = int(layer[0])
        filters_n = int(layer[1])
        # In case of invalid parameters return the previous layer
        if filter_h <= 0 or filter_w <= 0 or filters_n <= 0:
            return prev_layer

        # Create the convolutional layer
        conv = tf.layers.conv2d(prev_layer,
                                filters=filters_n,
                                kernel_size=(filter_h, filter_w),
                                strides=1,
                                name='conv'+c_l,
                                activation=tf.nn.relu6,
                                kernel_initializer=x_i(),
                                bias_initializer=tf.zeros_initializer(),
                                trainable=self.trainable)

        self.current_layer += 1
        return conv
