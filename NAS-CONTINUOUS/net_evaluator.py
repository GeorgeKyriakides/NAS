# -*- coding: utf-8 -*-
"""
Network architecture evaluator
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from cifar_data import load_cifar10
from net_builder import NetBuilder
import numpy as np


class NetEvaluator(object):
    # Defaults
    search_epochs = 3
    batch_size = 1000

    def __init__(self, actions_no, trainable=True, dataset='mnist'):
        self.data = None
        self.shape = None
        self.baseline = None
        # Configure the parameters according to the dataset
        if dataset == 'mnist':
            self.data = input_data.read_data_sets(
                "MNIST_data/", one_hot=True, reshape=False)
            self.shape = (None, 28, 28, 1)
            self.baseline = 0.95
        elif dataset == 'cifar10':
            self.data = load_cifar10(reshape=False)
            self.shape = (None, 32, 32, 3)
            self.baseline = 0.5
        self.batch_size = NetEvaluator.batch_size
         # Create a network builder
        self.builder = NetBuilder(self.shape, actions_no, trainable)
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
                try:
                    # Build the model
                    model = self.builder.build_net(state)
                    with tf.Session() as sess:
                        initer = tf.global_variables_initializer()
                        sess.run(initer)
                        # Train for given epochs
                        for epoch in range(epochs):
                            instances = self.data.train.num_examples
                            # Iterate through all batches
                            for s in range(instances//self.batch_size):
                                batch_x, batch_y = \
                                    self.data.train.next_batch(self.batch_size)
                                t, l, a = sess.run([model.train_op, model.loss_op,
                                                    model.accuracy], feed_dict={
                                    model.input_t: batch_x,
                                    model.target_t: batch_y})

                        # Evaluate on the test set
                        model.set_training(False)
                        acc, loss = 0, 0
                        steps = 10000//self.batch_size
                        for i in range(steps):
                            batch_x, batch_y = self.data.test.next_batch(100)
                            loss_, acc_ = sess.run([model.loss_op, model.accuracy],
                                                   feed_dict={model.input_t: batch_x,
                                                              model.target_t: batch_y})
                            acc += acc_
                            loss += loss_
                        acc /= steps
                        loss /= steps
                        model.set_training(True)
                        del model
                # If the GPU's memory is exhausted return 0 accuracy
                except tf.errors.ResourceExhaustedError:
                    acc = 0

        return acc
