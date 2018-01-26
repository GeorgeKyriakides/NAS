# -*- coding: utf-8 -*-
from enum import Enum
from mpi4py import MPI
import numpy as np
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MAX_DEPTH = 3
gamma = 0.995
EPS = 0.0
EPS_RED_FACTOR = 1.0
BATCH_SIZE = MAX_DEPTH*4
TRAINING_EPOCHS = 1
MAX_T = 4000

DEBUG = True


class tags(Enum):
    READY, DONE, EXIT, START = range(4)


class Worker(object):

    def __init__(self, sess, state_len, actions_no, max_depth, weights, workers_no):
        from ac_net import ACNet
        from net_evaluator import NetEvaluator
        global BATCH_SIZE
        self.processes = workers_no+1

        self.state_len = state_len
        self.actions_no = actions_no
        self.eps = EPS
        self.evaluator = NetEvaluator(trainable=True)
        self.actions_no = actions_no
        self.state = np.zeros(state_len)
        self.state[0] = 1

        self.max_depth = max_depth
        self.prev_acc = 0

        self.current_max_depth = max_depth
        self.old_weights = weights
        self.grads = []
        self.samples = []
        self.memory = dict()
        self.ac_net = ACNet(sess, self.state_len, self.actions_no, 'worker')
        self.ac_net.set_weights(self.old_weights)

    def update_ac_weights(self, weights):
        self.old_weights = weights
        if self.ac_net is not None:
            self.ac_net.set_weights(weights)

    def get_grads(self):
        return self.grads

    def calculate_gradients(self):
        grads = []
        weights = self.ac_net.get_weights()
        for i in range(len(weights)):
            grads.append(weights[i]-self.old_weights[i])
        return grads

    def fetch_from_memory(self, state):
        if state in self.memory:
            return self.memory[state]
        else:
            return None

    def add_to_memory(self, state, acc):
        self.memory[state] = acc

    def play(self):
        self.state = np.zeros(len(self.state))
        self.state[0] = 1
        self.prev_acc = self.evaluator.baseline
        del self.model
        self.model = None
        t_start = self.t

        episode_flag = True
        while episode_flag:

            policy, value = self.ac_net.predict(
                self.state.reshape(1, self.state_len))
            policy = policy[0]

            action = np.argmax(policy)
            reward, new_state = self.perform_action(action)
            self.state = new_state
            self.t += 1
            if self.t-t_start >= self.current_max_depth:
                episode_flag = False

        return self.prev_acc, self.state

    def run(self):

        self.grads = []
        self.t = 1
        self.episodes = 0
        self.samples = []
        self.eps = self.eps*EPS_RED_FACTOR

        while self.t <= BATCH_SIZE:

            self.state = np.zeros(len(self.state))
            self.state[0] = 1
            self.prev_acc = self.evaluator.baseline
            del self.model

            self.model = None

            t_start = self.t
            s_buffer = []
            r_buffer = []
            a_buffer = []

            episode_flag = True
            while episode_flag:

                policy, value = self.ac_net.predict(
                    self.state.reshape(1, self.state_len))

                policy = policy[0]
                value = value[0]
                action = np.random.choice(self.actions_no, p=policy)
                if np.random.uniform() < self.eps:
                    action = np.random.choice(self.actions_no)

                reward, new_state = self.perform_action(action)

                s_buffer.append(self.state)
                r_buffer.append(reward)
                a_buffer.append(action)

                self.state = new_state
                self.t += 1
                self.print_episode(policy, action, value, reward)
                if self.t-t_start >= self.current_max_depth:
                    episode_flag = False

            self.episodes += 1

            R = 0.0
            rev_rewards = []
            counter = 0
            for r in reversed(r_buffer):
                if counter == self.current_max_depth:
                    counter = 0
                    R = 0
                R = R * gamma + r
                rev_rewards.append(R)

            for reward, state,  action in zip(rev_rewards, reversed(s_buffer),
                                              reversed(a_buffer)):
                self.samples.append((state, action, reward))

            np.random.shuffle(self.samples)

            # Transfrom to column vectors
            state, action, reward = list(map(np.array, zip(*self.samples)))
            v_l, p_l, e, g_n, v_n, grads = self.ac_net.fit(
                state, action, reward)
            self.samples = []

            for i in range(len(grads)):
                if len(self.grads) == i:
                    self.grads.append(grads[i])
                else:
                    self.grads[i] = self.grads[i] + grads[i]

        if self.current_max_depth < self.max_depth:
            self.current_max_depth += 1

        return self.prev_acc, self.state
        # return self.play()

    def print_episode(self, policy, action, value,
                      reward):
        if DEBUG:
            print('Policy :\n',
                  np.array2string(policy, precision=3))
            print('Action :\n', action)
            print('Value  :\n', np.array2string(value, precision=3))
            print('State :', self.state)
            print('Reward : %.3f' % reward, 'Accuracy : %.3f' % self.prev_acc)

    def perform_action(self, action):
        # Get new state
        new_state = self.update_state(action)
        # Expand model and evaluate
        acc = self.fetch_from_memory(str(new_state))
        if acc is None:
            acc = self.evaluator.evaluate_model(new_state, epochs=TRAINING_EPOCHS)
            self.add_to_memory(str(new_state), acc)
        # Get the reward
        reward = acc-self.prev_acc

        self.prev_acc = acc
        return reward, new_state

    def update_state(self, action, old_state=None):
        '''
            Update the state, based on the action taken
        '''

        if old_state is None:
            old_state = np.copy(self.state)

        new_state = np.copy(old_state)

        # If we added a layer
        if action != 0:
            onehot_action = np.zeros(self.actions_no-1)
            onehot_action[action-1] = 1
            index = 1
            for depth in range(self.max_depth):
                start = depth*(self.actions_no-1)+1
                actives = 0
                for i in range(self.actions_no-1):
                    actives += old_state[start+i]
                if actives == 0:
                    index = start
                    break
            for i in range(self.actions_no-1):
                new_state[index+i] = onehot_action[i]
        return new_state


class Master(object):

    def __init__(self, sess, state_len, actions_no, workers_no):
        from ac_net import ACNet
        self.T = 0
        self.processes = workers_no+1
        self.ac_net = ACNet(sess, state_len, actions_no, 'Master')


def main():
    import session_manager

    from net_builder import NetBuilder
    builder = NetBuilder((1,))
    actions = builder.actions
    actions_no = len(actions)

    state_len = MAX_DEPTH*(actions_no-1)+1

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
        m = Master(sess, state_len, actions_no, workers_no=workers_no)
        weights_buff = m.ac_net.get_weights()
        weights_size = len(weights_buff)

    weights_size = comm.bcast(weights_size)
    if my_rank > 0:
        for i in range(weights_size):
            weights_buff.append([])
    for i in range(weights_size):
        tmp_weights = 0
        if my_rank == 0:
            tmp_weights = weights_buff[i].eval(sess)
        weights_buff[i] = comm.bcast(tmp_weights)
    worker = Worker(sess, state_len, actions_no, max_depth=MAX_DEPTH,
                    weights=weights_buff, workers_no=workers_no)
    worker.memory = comm.bcast(worker.memory)
    comm.Barrier()

    t = my_rank
    if my_rank == 0:
        import my_tools
        import os

        script_name = os.path.basename(__file__)
        script_name = script_name.replace('.py', '')
        logger = my_tools.ProgressLogger(script_name, actions)

    while(t < MAX_T):
        if my_rank == 0:
            # If it is ran inside spyder
            if (workers_no == 0):
                acc, state = worker.run()
                grads = worker.get_grads()
                eps = worker.eps
                logger.update(state, acc, t)
                logger.log(state, acc, t, eps, str(0))
                weights_buff = worker.grads
                m.ac_net.update_with_grads(weights_buff)
                t += 1

            else:

                # Gradients exchange
                for _ in range(workers_no):
                    grads = None
                    grads = comm.recv(status=status)
                    rnk = status.Get_source()
                    acc = comm.recv(status=status, source=rnk)
                    state = comm.recv(status=status, source=rnk)
                    eps = comm.recv(status=status, source=rnk)
                    logger.update(state, acc, t)
                    logger.log(state, acc, t, eps, str(rnk))
                    # Weights update
                    m.ac_net.update_with_grads(grads)

                    t += 1

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
        t += my_rank

    if my_rank == 0:
        print('--Done--')


if __name__ == "__main__":
    print('Start')
    main()
