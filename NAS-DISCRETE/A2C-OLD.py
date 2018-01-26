# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:04:17 2017

@author: G30
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 18:48:12 2017

@author: G30
"""


from mpi4py import MPI
import gc
import numpy as np
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
max_depth = 6
return_steps = 5
gamma = 0.995
verbose = -1
eps = 0.0
eps_best_sample_train = 0.00
red_factor = 1.0
batch_size = max_depth*4
eval_epochs = 1
max_T = 4000

DEBUG = True
PRINT_WEIGHTS = False
PRINT_MEMORY = False
POPULATE_MEMORY = False


def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


tags = enum('READY', 'DONE', 'EXIT', 'START')

trainer = tf.train.AdamOptimizer(learning_rate=1e-4)


class Worker(object):

    def __init__(self, sess, state_len, actions_no, t_max, weights, workers_no):
        from ac_net import ACNet
        #from net_builder import NetBuilder
        from net_evaluator import NetEvaluator
        global batch_size, trainer
        self.processes = workers_no+1

        self.state_len = state_len
        self.actions_no = actions_no
        self.eps = eps
        self.evaluator = NetEvaluator(trainable=True)
        self.builder = None
        self.d_theta = 0
        self.d_theta_v = 0
        self.alive = False
        self.actions_no = actions_no
        self.state = np.zeros(state_len)
        self.state[0] = 1
        self.state_len = state_len
        self.t_max = t_max  # max depth

        self.sample_index = []
        self.prev_acc = 0  # self.evaluator.baseline
        self.model = None

        self.current_max_depth = t_max  # 1
        self.old_weights = weights
        self.grads = []
        self.ac_net = None
        self.samples = []
        self.best_samples = []
        self.best_reward = -1000
        # self.replay_length=replay_length
        self.memory = dict()
        self.ac_net = ACNet(sess, self.state_len,
                            self.actions_no, 'worker', trainer)
        self.ac_net.set_weights(self.old_weights)

    '''
        !!!!!!!!!!!DEBUGGING ONLY!!!!!!!!!!!!!!!
    '''

    def populate_memory(self):
        self.max_acc = 0
        self.best_state = 'None'
        np.random.seed(89461514)
        state = np.zeros(self.state_len)
        state[0] = 1
        # State 00000000...0
        self.memory[str(state)] = 0  # self.evaluator.baseline
        old_acc = 0  # self.evaluator.baseline
        self.poss_act(state, 0, old_acc)
        '''
        rev = dict()
        for key in self.memory.keys():
            rev[self.memory[key]]=key
        for key in sorted(rev.keys()):
            print('%.3f'%key,rev[key])
        '''
        print('ACC:  %.3f' % self.max_acc, end=' ')
        print(str(self.best_state))

    def poss_act(self, state, depth, old_acc):
        original_sate = state.copy()

        self.add_to_memory(str(original_sate), old_acc)
        if depth == max_depth-1:

            for action in range(1, self.actions_no):
                new_state = self.update_state(action, original_sate)
                new_acc = old_acc+np.random.lognormal(0, 0.1)-0.05
                if new_acc > self.max_acc:
                    self.max_acc = new_acc
                    self.best_state = new_state
                self.add_to_memory(str(new_state), new_acc)
        else:
            new_depth = depth+1
            for action in range(1, self.actions_no):
                new_state = self.update_state(action, original_sate)
                new_acc = old_acc+np.random.lognormal(0, 0.1)-0.05
                if new_acc > self.max_acc:
                    self.max_acc = new_acc
                    self.best_state = new_state
                self.add_to_memory(str(new_state), new_acc)
                if np.sum(new_state) == 1:
                    print(original_sate)
                    print(action)
                self.poss_act(new_state, new_depth, new_acc)

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
        prev_trainable = self.evaluator.builder.trainable
        self.evaluator.builder.trainable = True
        self.state = np.zeros(len(self.state))
        self.state[0] = 1
        self.prev_acc = 0  # self.evaluator.baseline
        del self.model
        # gc.collect()
        self.model = None
        self.d_theta = 0
        self.d_theta_v = 0
        self.alive = True
        t_start = self.t
        s_buffer = []
        r_buffer = []
        a_buffer = []
        v_buffer = []
        mean_v = 0
        R = 0
        episode_flag = True
        while episode_flag:
            '''
            policy,value=self.ac_net.predict_on_batch([
                    self.state.reshape(1,self.state_len),
                    np.array([0])#,
                    #np.array([0.0])
                    ])
            '''
            policy, value = self.ac_net.predict(
                self.state.reshape(1, self.state_len))

            policy = policy[0]
            value = value[0]
            mean_v += value[0]

            # action=np.argmax(policy)
            action = np.argmax(policy)
            reward, new_state = self.perform_action(action)

            s_buffer.append(self.state)
            r_buffer.append(reward)
            a_buffer.append(action)
            v_buffer.append(value)
            R = reward+gamma*R
            self.state = new_state
            self.t += 1
            # T update by master!!!!!!!!
            # if self.t-t_start>=self.t_max:# or self.state[2]==1:# or self.state[5]==1:
            # or self.state[2]==1:# or self.state[5]==1:
            if self.t-t_start >= self.current_max_depth:
                episode_flag = False

        self.evaluator.builder.trainable = prev_trainable
        return self.prev_acc, self.state

    '''
    def start(self):
        p=threading.Thread(target=self.run)
        p.start()
    '''

    def run(self):

        self.grads = []
        self.t = 1
        R_mean = 0.0
        R_max = 0.0
        self.episodes = 0
        self.samples = []
        # Gather experiences
        # self.ac_net.clear_session()
        self.eps = self.eps*red_factor
        best_acc = 0
        best_state = None
        mean_v = 0
        while self.t <= batch_size:

            R = 0.0

            self.state = np.zeros(len(self.state))
            self.state[0] = 1
            self.prev_acc = 0  # self.evaluator.baseline
            del self.model
            # gc.collect()
            self.model = None
            self.d_theta = 0
            self.d_theta_v = 0
            self.alive = True
            t_start = self.t
            s_buffer = []
            r_buffer = []
            a_buffer = []
            v_buffer = []
            episode_flag = True
            while episode_flag:
                '''
                policy,value=self.ac_net.predict_on_batch([
                        self.state.reshape(1,self.state_len),
                        np.array([0])#,
                        #np.array([0.0])
                        ])
                '''
                policy, value = self.ac_net.predict(
                    self.state.reshape(1, self.state_len))

                policy = policy[0]
                value = value[0]
                mean_v += value[0]

                # action=np.argmax(policy)
                action = np.random.choice(self.actions_no, p=policy)
                if np.random.uniform() < self.eps:  # or len(self.samples)<self.replay_length:
                    action = np.random.choice(self.actions_no)
                if DEBUG:
                    print('Policy :', policy)
                    print('Action :', action)
                    print('Value  :', value)

                reward, new_state = self.perform_action(action)

                s_buffer.append(self.state)
                r_buffer.append(reward)
                a_buffer.append(action)
                v_buffer.append(value)
                R = reward+gamma*R
                self.state = new_state
                self.t += 1
                # T update by master!!!!!!!!
                # if self.t-t_start>=self.t_max:# or self.state[2]==1:# or self.state[5]==1:
                # or self.state[2]==1:# or self.state[5]==1:
                if self.t-t_start >= self.current_max_depth:
                    episode_flag = False

            if R_max < R:
                R_max = R
            if self.prev_acc > best_acc:
                best_acc = self.prev_acc
                best_state = self.state.copy()
            self.episodes += 1
            R_mean += R
            R = 0.0
            rev_rewards = []
            counter = 0
            for r in reversed(r_buffer):
                if counter == self.current_max_depth:
                    counter = 0
                    R = 0
                R = R * gamma + r
                rev_rewards.append(R)
            if DEBUG:
                print("Reversed rewards: ", rev_rewards)

            for reward, state,  value, action in zip(rev_rewards, reversed(s_buffer), reversed(v_buffer), reversed(a_buffer)):

                advantage = reward - value
                '''
                    print(state,end=' ')
                    print(action,end=' ')
                    print(advantage,end=' ')
                    print(value,end=' ')
                    print(reward)
                    '''

                str_rep = str(state)+str(action)
                '''
                    if str_rep not in self.sample_index:
                        self.sample_index.append(str_rep)
                        self.samples.append((state, action, advantage, reward))
                    else:
                        print(str_rep)
                    print('LEEEEEEEEEEENL'+ str(len(self.samples)))
                    '''
                self.samples.append((state, action, advantage, reward))

            if verbose > -1:
                print('sample added: '+str())

            # while len(self.samples)>mem_size:
                #del self.samples[0]
            np.random.shuffle(self.samples)
            # ret_samples=self.samples[:self.replay_length]
            ret_samples = self.samples
            # Transfrom to column vectors
            state, action, advantage, reward = list(
                map(np.array, zip(*ret_samples)))
            # MY REPLAY!!!!!!!!!!
            if np.mean(reward) >= self.best_reward:
                self.best_reward = np.mean(reward)
                self.best_samples = [state, action, advantage, reward]
            if np.random.uniform() < eps_best_sample_train:
                state_, action_, advantage_, reward_ = self.best_samples
                np.append(state, state_)
                np.append(action, action_)
                np.append(advantage, advantage_)
                np.append(reward, reward_)
            #--------------------------------------------------------------------
            # if len(self.samples)>=self.replay_length:
            v_l, p_l, e, g_n, v_n, grads = self.ac_net.fit(
                state, action, advantage, reward)
            self.samples = []

            for i in range(len(grads)):
                if len(self.grads) == i:
                    self.grads.append(grads[i])
                else:
                    self.grads[i] = self.grads[i] + grads[i]
        logger.warning('V_L: %.3f, P_L: %.3f, E: %.3f, G_N: %.3f, V_N: %.3f, ' % (
            v_l, p_l, e, g_n, v_n))
        # print('-'*50)
        #print('V_L: ',v_l,' P_L: ',p_l,' E: ',e, end=' ')
        # print('-'*50)
        #print('G_N: ',g_n,' V_N: ',v_n)
        # print('-'*50)

        #print(0.5 * v_l + p_l - e * 0.01)
        # print('-'*50)
        if self.current_max_depth < self.t_max:
            self.current_max_depth += 1
            # self.replay_length+=replay_length
        # Dirty gradients extraction
        # self.grads=grads#self.calculate_gradients()
        if DEBUG:
            print("Have %d samples from %d episodes, mean final reward: %.3f, max: %.3f, mean value: %.3f" % (
                len(self.samples), self.t, np.mean(reward), np.max(reward),
                mean_v/len(self.samples)))
        # DONE
        self.alive = False
        R_mean /= self.episodes

        # return np.mean(reward),best_state#self.prev_acc,self.state
        return self.play()

    def build_model(self):

        self.model = None
        self.t = 1

        # self.ac_net=ACNet(self.state_len,self.actions_no,process_no=self.processes)
        self.ac_net.set_weights(self.old_weights)
        self.d_theta = 0
        self.d_theta_v = 0
        self.alive = True
        t_start = self.t
        s_buffer = []
        r_buffer = []
        a_buffer = []
        v_buffer = []
        R = 0.0
        # Gather experiences
        while True:

            policy, value, loss = self.ac_net.predict_on_batch([
                self.state.reshape(1, self.state_len),
                np.array([0]),
                np.array([0.0])
            ])
            policy = policy[0]
            value = value[0]
            action = np.argmax(self.actions_no)
            reward, new_state = self.perform_action(action)

            s_buffer.append(self.state)
            r_buffer.append(reward)
            a_buffer.append(action)
            v_buffer.append(value)
            R = reward+gamma*R
            self.state = new_state
            self.t += 1
            # T update by master!!!!!!!!
            # or self.state[5]==1:
            if self.t-t_start >= self.t_max or self.state[2] == 1:
                return self.prev_acc

    def perform_action(self, action):
        # Get new state
        new_state = self.update_state(action)

        # Printing
        self.print_action(action)
        self.print_state(new_state)
        # Expand model and evaluate
        acc = self.fetch_from_memory(str(new_state))
        if acc is None:
            if PRINT_MEMORY:
                print(self.memory)
                print(new_state)
            acc = self.evaluator.evaluate_model(
                new_state, verbose_in=verbose, epochs=eval_epochs)
            self.add_to_memory(str(new_state), acc)
        # Get the reward
        reward = acc-self.prev_acc
        # reward=np.sign(acc-self.prev_acc)

        # reward=reward+reward**2
        self.prev_acc = acc
        return reward, new_state

    def update_state(self, action, old_state=None):
        '''
            Update the state, based on the action taken
        '''
        #from keras.utils import to_categorical

        if old_state is None:
            old_state = np.copy(self.state)

        new_state = np.copy(old_state)

        # If we added a layer
        if action != 0:
            onehot_action = np.zeros(self.actions_no-1)
            onehot_action[action-1] = 1
            index = 1
            for depth in range(self.t_max):
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

    def print_state(self, to_print=None):
        if verbose > -1:
            if to_print is None:
                to_print = self.state
            print('State:', end='')
            for x in to_print:
                print(int(x), end='')
            print('')

    def print_action(self, action):
        if verbose > -1:
            print('Action:'+str(self.builder.actions[action]))


class Master(object):

    def __init__(self, sess, state_len, actions_no, workers_no):
        from ac_net import ACNet
        global trainer
        self.T = 0
        self.processes = workers_no+1
        # Next, we build a very simple model.
        self.ac_net = ACNet(sess, state_len, actions_no, 'Master', trainer)
        # print(self.ac_net.model.summary())
        # self.worker=Worker(state_len,actions_no,t_max=2,weights=self.ac_net.get_weights())
        # self.worker2=Worker(state_len,actions_no,t_max=2,weights=self.ac_net.get_weights())


def main():
    global POPULATE_MEMORY
    #from net_builder import NetBuilder
    import session_manager
    import time

    if POPULATE_MEMORY:
        conv_filters = []  # [16,32,64]
        conv_kernels = []  # [1,2,3]
        conv_list = []
        for f in conv_filters:
            for k in conv_kernels:
                conv_list.append(str(f)+','+str(k))
        types = {
            'fc': [64, 128, 256, 512], 'dout': [0.1, 0.3, 0.5], 'conv2d': conv_list
        }
        actions = ['NONE-NONE']
        for key in types.keys():
            for parameter in types[key]:
                actions.append(str(key)+'-'+str(parameter))
    else:
        from net_builder import NetBuilder
        builder = NetBuilder((1,))
        actions = builder.actions
    actions_no = len(actions)

    state_len = max_depth*(actions_no-1)+1

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    status = MPI.Status()
    group = comm.Get_group()
    newgroup = group.Excl([0])
    work_comm = comm.Create(newgroup)
    np.random.seed()
    workers_no = comm.Get_size()-1
    print('Total processes: '+str(workers_no+1))

    print(my_rank)
    sess = session_manager.get_session(workers_no+1)

    comm.Barrier()
    weights_buff = []
    weights_size = 0
    if my_rank == 0:
        m = Master(sess, state_len, actions_no, workers_no=workers_no)
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
    worker = Worker(sess, state_len, actions_no, t_max=max_depth,
                    weights=weights_buff, workers_no=workers_no)
    worker.memory = comm.bcast(worker.memory)
    comm.Barrier()
    if my_rank == 0:
        if POPULATE_MEMORY:
            worker.populate_memory()
    worker.memory = comm.bcast(worker.memory)

    t = my_rank
    if my_rank == 0:
        import my_tools
        import os

        script_name = os.path.basename(__file__)
        script_name = script_name.replace('.py', '')
        # actions=NetBuilder((1,)).actions
        logger = my_tools.ProgressLogger(script_name, actions)

    while(t < max_T):
        if my_rank == 0:
            # If it is ran inside spyder
            if (workers_no == 0):
                acc, state = worker.run()
                grads = worker.get_grads()
                eps = worker.eps
                logger.update(state, acc, t)
                logger.log(state, acc, t, eps, str(0))
                weights_buff = worker.grads
                # print(weights_buff)
                # print(worker.ac_net.model.layers)
                # CAUTION: KERAS SESSION IS CLEARED BY WORKER CALL!!!!
                # m.ac_net.build_model()
                m.ac_net.update_with_grads(weights_buff)
                # print('-------------------------------------')
                # print(m.ac_net.model.layers)
                # weights_buff=m.ac_net.get_weights()
                # worker.update_ac_weights(weights_buff)
                t += 1
                #print('Done :',t)

            else:
                #print('AC Net weights distributed')
                # Gradients xchange
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
                    # weights_buff=m.ac_net.get_weights()
                    print(str(rnk)+'-Done '+str(t))
                    print('---------------------------')

                    # Weights xchange
                    # for i in range(weights_size):
                    #    comm.send(weights_buff[i].eval(sess),dest=rnk,tag=tags.READY)
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
            '''
            weights_buff=[]
            for i in range(weights_size):
                weights_buff.append([])
                weights_buff[i]=comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == tags.EXIT:
                break
            '''

        # Sync update
        weights_buff = []
        timestamp = time.strftime("%d_%m_%Y-%H_%M_%S")
        for i in range(weights_size):
            weights_buff.append([])

        if my_rank == 0:
            weights_buff = m.ac_net.get_weights()

            if PRINT_WEIGHTS:
                for i in range(len(weights_buff)):
                    np.savetxt('MASTER WEIGHTS'+str(rnk)+'_'+str(i) +
                               '_'+timestamp, (weights_buff[i].eval(sess)))
            for i in range(weights_size):
                weights_buff[i] = weights_buff[i].eval(sess)

        for i in range(weights_size):
            weights_buff[i] = comm.bcast(weights_buff[i])

        if PRINT_WEIGHTS and my_rank == 1:
            for i in range(len(weights_buff)):
                np.savetxt('WORKER MASTER WEIGHTS'+str(my_rank) +
                           '_'+str(i)+'_'+timestamp, (weights_buff[i]))

        worker.update_ac_weights(weights_buff)

        if PRINT_WEIGHTS and my_rank == 1:
            weights_buff = worker.ac_net.get_weights()
            for i in range(len(weights_buff)):
                np.savetxt('WORKER WEIGHTS'+str(my_rank)+'_'+str(i) +
                           '_'+timestamp, (weights_buff[i].eval(sess)))

        t = comm.bcast(t)
        t += my_rank

    if my_rank == 0:
        return worker
    '''
        for i in range(workers_no):
            grads=comm.recv(status=status)
            rnk = status.Get_source()
            comm.send(None,dest=rnk,tag=tags.EXIT)
        #Build final model
        weights_buff=m.ac_net.get_weights()
        w=Worker(sess,state_len,actions_no,t_max=max_depth,weights=weights_buff,workers_no=workers_no)
        print('Final acc: ')
        print(w.build_model())
    '''


if __name__ == "__main__":
    print('Start')
    worker = main()
    # worker.populate_memory()
