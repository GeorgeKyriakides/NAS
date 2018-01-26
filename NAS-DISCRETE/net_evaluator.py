# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 18:29:25 2017

@author: G30
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from net_builder import NetBuilder
import numpy as np


class NetEvaluator(object):

    search_epochs = 10
    baseline = 0.9
    batch_size = 512

    def __init__(self, trainable=True):

        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.batch_size = NetEvaluator.batch_size
        self.builder = NetBuilder((None, 784), trainable)
        self.exp_no = 0

    def evaluate_model(self, state, epochs=search_epochs):
        '''
            Train and evaluate on test set
        '''
        if np.sum(state) == 1:
            return NetEvaluator.baseline
        self.exp_no += 1
        container_name = 'experiment'+str(self.exp_no)
        with tf.Graph().as_default() as g:
            with g.container(container_name):
                # Build the model
                model = self.builder.build_net(state)
                max_acc = 0
                with tf.Session() as sess:
                    initer = tf.global_variables_initializer()
                    sess.run(initer)
                    # Train for given epochs
                    for epoch in range(epochs+1):
                        instances = self.mnist.train.num_examples
                        # Iterate through all batches
                        for s in range(instances//self.batch_size):
                            batch_x, batch_y = \
                                self.mnist.train.next_batch(self.batch_size)
                            t, l, a = sess.run([model.train_op, model.loss_op,
                                                model.accuracy], feed_dict={
                                model.input_t: batch_x,
                                model.target_t: batch_y})

                        model.set_training(False)
                        batch_x, batch_y = self.mnist.test.next_batch(10000)
                        loss, acc = sess.run([model.loss_op, model.accuracy],
                                             feed_dict={model.input_t: batch_x,
                                                        model.target_t: batch_y})

                        if acc > max_acc:
                            max_acc = acc
                        model.set_training(True)
        return max_acc
