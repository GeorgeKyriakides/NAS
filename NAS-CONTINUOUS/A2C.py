# -*- coding: utf-8 -*-
from mpi4py import MPI
import numpy as np
import os
from enum import Enum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MAX_DEPTH = 2


# Discount factor
gamma = 0.995
# EPSilon greedy
EPS = 0.00
EPS_RED_FACTOR = 1.0

# Training epochs for the solution architectures
TRAIN_EPOCHS = 10

return_episode = True
MAX_T = 4000

# Print results for each episode
DEBUG = True
# Plot the netorks' accuracies
LIVE_PLOT = False

ACTIONS_BOUNDS = np.array([[1, 5], [1, 128]], dtype='float32')
ACTIONS_BOUNDS = ACTIONS_BOUNDS.transpose()
ACTIONS_NO = len(ACTIONS_BOUNDS[0])

STARTING_STATE = np.array([10.]*ACTIONS_NO)
STARTING_STATE = np.append(STARTING_STATE, np.zeros((MAX_DEPTH)*ACTIONS_NO))



class tags(Enum):
    READY, DONE, EXIT, START = range(4)

class Worker(object):

    def __init__(self, sess, state_len, actions_no, actions_bounds, max_depth,
                 weights, workers_no, dataset, trainable):
        from ac_net import ACNet
        from net_evaluator import NetEvaluator


        self.processes = workers_no+1
        self.actions_no = actions_no
        self.eps = EPS
        self.evaluator = NetEvaluator(
            ACTIONS_NO, trainable=trainable, dataset=dataset)
        self.state = STARTING_STATE.copy()
        self.state_len = state_len
        self.max_depth = max_depth
        self.t = 1
        self.prev_acc = self.evaluator.baseline
        self.model = None
        self.current_max_depth = self.max_depth
        self.old_weights = weights
        self.grads = []
        self.samples = []
        self.best_samples = []
        self.best_reward = -1000
        self.memory = dict()
        self.ac_net = ACNet(sess, self.state_len,
                            self.actions_no, actions_bounds, 'worker')
        self.ac_net.set_weights(self.old_weights)


    def update_ac_weights(self, weights):
        self.old_weights = weights
        if self.ac_net is not None:
            self.ac_net.set_weights(weights)

    def get_grads(self):
        return self.grads


    def fetch_from_memory(self, state):
        state_repr = state.copy()
        for i in range(len(state)):
            if i % ACTIONS_NO == ACTIONS_NO-1:
                state_repr[i] = np.round(state_repr[i], 1)
            else:
                state_repr[i] = np.round(state_repr[i], 0)
        state_repr = str(state_repr)
        if state_repr in self.memory:
            return self.memory[state_repr]
        else:
            return None

    def add_to_memory(self, state, acc):
        state_repr = state.copy()
        for i in range(len(state)):
            if i % ACTIONS_NO == ACTIONS_NO-1:
                state_repr[i] = np.round(state_repr[i], 1)
            else:
                state_repr[i] = np.round(state_repr[i], 0)
        state_repr = str(state_repr)
        self.memory[state_repr] = acc

    def play(self):
        prev_trainable = self.evaluator.builder.trainable
        self.evaluator.builder.trainable = True
        self.state = STARTING_STATE.copy()
        self.prev_acc = 0
        t_start = self.t
        episode_flag = True
        self.current_layer = 0
        while episode_flag:

            action, policy_mean, policy_sigma, value = self.ac_net.predict(
                self.state.reshape(1, self.state_len//self.actions_no, self.actions_no))

            value = value[(self.current_layer)]
            reward, new_state = self.perform_action(action)

            self.state = new_state
            self.t += 1
            self.current_layer += 1
            if self.t-t_start >= self.current_max_depth:
                episode_flag = False

        self.evaluator.builder.trainable = prev_trainable
        return self.prev_acc, self.state


    def run(self):

        self.grads = []
        self.samples = []
        t_start = self.t
        # Gather experiences
        self.eps = self.eps*EPS_RED_FACTOR

        while self.t-t_start < self.max_depth:
            self.current_layer = 0
            R = 0.0
            self.state = STARTING_STATE.copy()
            self.prev_acc = self.evaluator.baseline
            del self.model

            self.model = None
            self.d_theta = 0
            self.d_theta_v = 0
            self.alive = True

            s_buffer = []
            r_buffer = []
            a_buffer = []
            v_buffer = []

            episode_flag = True
            while episode_flag:

                action, policy_mean, policy_sigma, value = self.ac_net.predict(
                    self.state.reshape(1, self.state_len//self.actions_no, self.actions_no))

                action = action[(self.current_layer)]

                if np.random.uniform() < self.eps:
                    action = (np.random.uniform() *
                                 (ACTIONS_BOUNDS[1]-ACTIONS_BOUNDS[0]))//1

                value = value[(self.current_layer)]


                reward, new_state = self.perform_action(action)

                r_buffer.extend(([reward]))
                a_buffer.append(([action]))
                v_buffer.append(([value]))
                R = reward+gamma*R
                self.state = new_state
                self.t += 1
                self.current_layer += 1

                self.print_episode(policy_mean, policy_sigma,
                                   action, value, reward)
                if self.current_layer >= self.current_max_depth:
                    episode_flag = False

            # Kill grads
            r_buffer.extend(([0]))
            a_buffer.append(([policy_mean[-1]]))
            v_buffer.append(([0]))
            # Add state
            s_buffer.append(self.state.reshape(
                1, self.state_len//self.actions_no, self.actions_no))

            R = 0.0
            rev_rewards = []
            for r in reversed(r_buffer):
                R = R * gamma + r
                rev_rewards.append(R)


            reward = rev_rewards.reverse()
            reward = np.array(rev_rewards).reshape((-1, 1))
            action = np.array(a_buffer).reshape((-1, self.actions_no))
            state = self.state

            self.samples.append((self.state, action, reward))



        np.random.shuffle(self.samples)

        # Transfrom to column vectors
        state, action, reward = list(map(np.array, zip(*self.samples)))


        v_l, p_l, e, grads = self.ac_net.fit(state, action, reward)
        self.samples = []
        self.grads = grads

        if self.current_max_depth < self.max_depth and self.t > 100:
            self.current_max_depth += 1


        self.grads = self.ac_net.get_grads()
        if return_episode:
            return self.prev_acc, self.state
        else:
            return self.play()


    def perform_action(self, action, search_mem=True):

        def get_acc(new_state):

            return self.evaluator.evaluate_model(new_state, epochs=TRAIN_EPOCHS)

        # Get new state
        new_state = self.update_state(action)


        # Build the model and evaluate
        acc = self.fetch_from_memory(new_state)
        if not search_mem:
            acc = get_acc(new_state)
        else:
            if acc is None:
                acc = get_acc(new_state)
                self.add_to_memory(new_state, acc)
        # Get the reward
        reward = (acc-self.prev_acc)
        self.prev_acc = acc
        return reward, new_state

    def update_state(self, action, old_state=None):
        '''
            Update the state, based on the action taken
        '''

        if old_state is None:
            old_state = np.copy(self.state)

        new_state = np.copy(old_state)
        index = (self.current_layer+1)*ACTIONS_NO


        for i in range(self.actions_no):
            new_state[index+i] = action[i]

        return new_state

    def print_episode(self, policy_mean, policy_sigma, action, value,
                      reward):
        if DEBUG:
            print('Policy_mean :\n',
                          np.array2string(policy_mean, precision=3))
            print('Policy_sigma :\n',
                  np.array2string(policy_sigma, precision=3))
            print('Action :\n', action)
            print('Value  :\n', np.array2string(value, precision=3))
            print('Layer :', self.current_layer)
            print('State :',self.state)
            print('Reward : %.3f' % reward, 'Accuracy : %.3f' % self.prev_acc)



class Master(object):

    def __init__(self, sess, state_len, actions_no, action_bounds, workers_no):
        from ac_net import ACNet
        self.T = 0
        self.processes = workers_no+1
        # Next, we build a very simple model.
        self.ac_net = ACNet(sess, state_len, actions_no,
                            action_bounds, 'Master')



def main(dataset, trainable):


    import session_manager


    actions_no = ACTIONS_NO
    state_len = len(STARTING_STATE)
    actions_bounds = ACTIONS_BOUNDS.copy()

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    status = MPI.Status()
    group = comm.Get_group()
    newgroup = group.Excl([0])
    work_comm = comm.Create(newgroup)
    np.random.seed()
    workers_no = comm.Get_size()-1
    if my_rank == 0:
        print('Total processes: '+str(workers_no+1))
    sess = session_manager.get_session(workers_no+1)

    comm.Barrier()
    weights_buff = []
    weights_size = 0
    if my_rank == 0:
        m = Master(sess, state_len, actions_no,
                   actions_bounds, workers_no=workers_no)
        weights_buff = m.ac_net.get_weights()
        print('AC Net generated')
        weights_size = len(weights_buff)

    weights_size = comm.bcast(weights_size)
    if my_rank > 0:
        for i in range(weights_size):
            weights_buff.append([])
    for i in range(weights_size):
        tmp = 0
        if my_rank == 0:
            tmp = weights_buff[i].eval(sess)
        weights_buff[i] = comm.bcast(tmp)
    worker = Worker(sess, state_len, actions_no, actions_bounds,
                    MAX_DEPTH, weights_buff,
                    workers_no, dataset, trainable)

    comm.Barrier()



    t = my_rank
    if my_rank == 0:
        import my_tools
        import os

        script_name = os.path.basename(__file__)
        script_name = script_name.replace('.py', '')
        trainable_tag = 'Trainable' if trainable else 'Nontrainable'
        log_tag = dataset+'_'+trainable_tag
        logger = my_tools.ProgressLogger(script_name,
                                         log_tag, None,
                                         representation=None,
                                         live_plot=LIVE_PLOT)

    while(t < MAX_T):
        if my_rank == 0:

            # If it is ran inside spyder
            if (workers_no == 0):
                rnk = 0
                acc, state = worker.run()
                grads = worker.get_grads()
                EPS = worker.eps
                logger.update(state, acc, t)
                logger.log(state, acc, t, EPS, str(0))
                t += 1

            else:
                # Gradients xchange
                avg_grads = dict()
                avg_grads['p'] = []
                avg_grads['v'] = []
                for _ in range(workers_no):
                    grads = comm.recv(status=status)
                    rnk = status.Get_source()
                    acc = comm.recv(status=status, source=rnk)
                    state = comm.recv(status=status, source=rnk)
                    EPS = comm.recv(status=status, source=rnk)
                    logger.update(state, acc, t)
                    logger.log(state, acc, t, EPS, str(rnk))
                    t += 1
                    if len(avg_grads['p']) == 0:
                        for i in range(len(grads['p'])):
                            avg_grads['p'].append(grads['p'][i])
                        for i in range(len(grads['v'])):
                            avg_grads['v'].append(grads['v'][i])
                    else:
                        for i in range(len(grads['p'])):
                            avg_grads['p'][i] += grads['p'][i]
                        for i in range(len(grads['v'])):
                            avg_grads['v'][i] += grads['v'][i]
                    # Weights update

                for i in range(len(avg_grads['v'])):
                    avg_grads['v'][i] /= workers_no
                for i in range(len(avg_grads['p'])):
                    avg_grads['p'][i] /= workers_no

                m.ac_net.update_with_grads(avg_grads)

        else:

            work_comm.Barrier()
            acc, state = worker.run()
            # Gradients xchange
            grads = worker.get_grads()
            comm.send(grads, dest=0)
            comm.send(acc, dest=0)
            comm.send(state, dest=0)
            comm.send(worker.eps, dest=0)
            # Weights xchange

        if workers_no > 0:
            # Sync update
            weights_buff = []
            for i in range(weights_size):
                weights_buff.append([])

            if my_rank == 0:
                weights_buff = m.ac_net.get_weights()

                for i in range(weights_size):
                    weights_buff[i] = weights_buff[i].eval(sess)

            for i in range(weights_size):
                weights_buff[i] = comm.bcast(weights_buff[i])

            worker.update_ac_weights(weights_buff)
            t = comm.bcast(t)

    if my_rank == 0:
        print('--DONE--')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='A2C')

    parser.add_argument('dataset', type=str,
                        help='Dataset: minst or cifar10')

    parser.add_argument('-t',  metavar='--trainable', type=bool,
                        default=False,const=True,nargs='?',
                        help='Make all the layers trainable')


    args = parser.parse_args()

    if args.dataset not in ['mnist', 'cifar10']:
        print('Please select a valid dataset')
    else:
        dataset = args.dataset
        trainable = args.t
        print('Start')
        main(dataset, trainable)
