# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:58:32 2017

@author: G30
"""

import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


timestamp = time.strftime("%d_%m_%Y-%H_%M_%S")


class ProgressLogger(object):

    def __init__(self, name, tag,
                 actions, representation='one_hot', live_plot=False):
        self.representation = representation
        self.name = name
        self.tag = tag
        self.actions = actions
        self.start_time = time.time()
        self.best = 0.0
        self.best_epochs = 0
        self.model_str = ''
        self.best_str = ''
        self.log_file = "tests/"+timestamp+'-'+self.name+'_'+self.tag+'.csv'
        self.fieldnames = ['BestLayers', 'Best', 'BestEpochs',
                           'CurrentLayers', 'Current', 'CurrentEpochs',
                           'TotalTime', 'Epsilon', 'Tag']

        self.live_plot = live_plot
        self.__create_log()
        if self.live_plot:
            self.x_counter = 0

    def animate(self):
        a = pd.read_csv(self.log_file)
        y = list(a.Current)
        x = list(a.CurrentEpochs)
        img = plt.plot(x, y)
        plt.savefig('fig.png')
        plt.close()

    def log(self, state, acc, epochs, epsilon, tag=None):
        elapsed_time = time.time() - self.start_time
        current_layers = self.get_representation(state)
        with open(self.log_file, 'a') as csvfile:
            self.writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames,
                                         lineterminator='\n')
            self.writer.writerow({'BestLayers': self.best_str,
                                  'Best': self.best,
                                  'BestEpochs': self.best_epochs,
                                  'CurrentLayers': current_layers,
                                  'Current': acc,
                                  'CurrentEpochs': epochs,
                                  'TotalTime': '%.2f' % elapsed_time,
                                  'Epsilon': epsilon,
                                  'Tag': tag})

    def repr_from_state2d(self, state):
        representation = ''
        for i in range(len(state[0])):
            if state[0][i] != 0:
                representation = representation + \
                    'X'+str(self.actions[state[0][i]])
            else:
                representation = representation + \
                    'X'+str(self.actions[state[1][i]])
        representation = representation[1:]
        return representation

    def repr_from_state1d(self, state):
        representation = ''
        for i in range(len(state)):
            representation = representation+'X'+str(int(state[i]))
        representation = representation[1:]
        return representation

    def repr_from_onehot(self, state):
        representation = ''
        # max_depth=int(len(state)/(len(self.actions)-1))
        if state is not None:
            state = state[1:]
            for i in range(len(state)):
                if state[i] == 1:
                    index = i % (len(self.actions)-1)
                    representation = representation + \
                        'X'+str(self.actions[index+1])
            representation = representation[1:]
        return representation

    def get_representation(self, state):
        rep = ''
        if self.representation == 'one_hot':
            rep = self.repr_from_onehot(state)
        elif self.representation == 'state2d':
            rep = self.repr_from_state2d(state)
        elif self.representation == 'state1d':
            rep = self.repr_from_state1d(state)
        else:
            if state is not None:
                rep = np.array2string(state, precision=1, separator=',')
                rep = rep.replace(' ', '').replace('\n', '')
                rep = rep.replace('.,', 'X').replace(',', 'X')
        return rep

    def update(self, state, acc, epochs):
        if acc > self.best:
            print('%.3f' % acc, '%.3f' % self.best)  # , end=' ')
            # print(state)
            self.best = acc
            self.best_str = self.get_representation(state)
            self.best_epochs = epochs
        if self.live_plot:
            self.x_counter += 1
            if(self.x_counter % 50 == 0):
                self.animate()

    def __create_log(self):
        from shutil import copyfile
        copyfile(self.name+'.py', "tests/"+timestamp+"-"+self.name+'.py')
        with open(self.log_file, 'w') as csvfile:
            self.writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames,
                                         lineterminator='\n')
            self.writer.writeheader()
