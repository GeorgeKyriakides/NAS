# -*- coding: utf-8 -*-
"""
    Double Deep Q-Learning Networks
"""

import numpy as np
import keras
import random
import os
import my_tools

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers.advanced_activations import LeakyReLU

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
script_name = os.path.basename(__file__)
script_name = script_name.replace('.py', '')

# Possible actions
dense = [128, 256, 384, 512, 640, 768, 896, 1024]
dropouts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# First action leaves the network intact
actions = [0]

actions.extend(dense)
actions.extend(dropouts)
# Create the log
logger = my_tools.ProgressLogger(script_name, actions)

max_depth = 4
start_depth = 1


max_memory = 2000
vals_memory = dict()
memory = []

gamma = 1
# depth+1 for the final state, len-1 because the first action
# is "stay"(no layers added)
states_no = (max_depth+1)*(len(actions)-1)
onehot_actions = keras.utils.to_categorical([x for x in range(len(actions)-1)])
states = np.zeros(states_no, dtype='int')

# Epsilon is bounded to [1.0,0.01]
epsilon = 1.0
epsilon_min = 0.01

epsilon_decay = 0.995

episodes = 200
baseline = 0.9
search_epochs = 3


def get_data():
    '''
        Get MNIST data
    '''
    num_classes = 10

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)


def evaluate_model(model_in, x_train, y_train, x_test, y_test, epochs=search_epochs, verbose_in=0):
    '''
        Train and evaluate on test set
    '''
    batch_size = 128

    model_str = ''
    model = model_in
    # Add classification layer
    model.add(Dense(10, activation='softmax'))
    # Print the model's layers
    for layer in model.layers:
        model_str = model_str+str(layer.input_shape)
    print(model_str, end=' ')
    # Compile and train
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose_in,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    # Remove classification layer
    model.pop()
    return score[1]


def generate_model():
    '''
        Generate random model
    '''

    # Avoid action 0
    action = np.random.randint(1, len(actions))
    model = Sequential()
    model.add(Dense(actions[action], activation='relu', input_shape=(784,)))

    return model, action


def expand_model(model, action):
    '''
        Act upon the architecture, adding a layer
    '''
    if model is None:
        model = Sequential()
        if actions[action] in dropouts:
            model.add(Dropout(actions[action], input_shape=(784,)))
        elif actions[action] in dense:
            model.add(
                Dense(actions[action], activation='relu', input_shape=(784,)))
        else:
            model = None
        return model

    # Terminal action, leaves the layers intact
    if actions[action] == 0:
        return model
    elif actions[action] in dense:
        model.add(Dense(actions[action], activation='relu'))
    else:
        model.add(Dropout(actions[action]))
    return model


def remember(states, action, acc, prev_acc, next_state):
    '''
        Reward is difference in accuracy
    '''
    reward = (acc-prev_acc)
    memory.append([states, action, reward, next_state])
    vals_memory[logger.repr_from_onehot(states)+','+str(action)] = acc
    if len(memory) >= max_memory:
        del memory[0]


def replay(ql, sz):
    '''
        Sample from memory and train the net
    '''
    global epsilon
    minibatch = random.sample(memory, sz)
    for state, action, reward, next_state in minibatch:
        state = state.reshape(1, states_no)
        next_state = next_state.reshape(1, states_no)
        target = reward + gamma*np.amax(qv.predict(next_state)[0])
        target_f = ql.predict(state)
        target_f[0][action] = target
        ql.fit(state, target_f, epochs=1, verbose=0)
    # Reduce epsilon
    if epsilon > epsilon_min:
        epsilon = epsilon*epsilon_decay
    update_q_target(ql, qv)


def train_q(ql):
    '''
        The RL
    '''
    (x_train, y_train), (x_test, y_test) = get_data()

    current_depth = start_depth

    for episode in range(episodes):
        if episode % 10 == 9:
            current_depth = min(current_depth+1, max_depth)
        prev_acc = baseline
        states = np.zeros(states_no, dtype='int')
        model = None

        for x in range(current_depth):
            print('Depth: '+str(x), end=' ')
            action = int(np.argmax(
                ql.predict(states.reshape(1, states_no))[0]
            ))
            if np.random.rand() <= epsilon:
                action = np.random.randint(len(actions))

            new_state = update_state(states, action)


            print('Action: ',action, end=' ')
            model = expand_model(model, action)

            if model is None:
                acc = baseline
                print('')
            elif logger.repr_from_onehot(states)+','+str(action) not in vals_memory:
                acc = evaluate_model(model, x_train, y_train, x_test, y_test)
                model.save_weights(
                    'weights_cache/'+logger.get_representation(states)+','+str(action))
                print(acc)
            else:
                acc = vals_memory[logger.repr_from_onehot(
                    states)+','+str(action)]
                model.load_weights(
                    'weights_cache/'+logger.get_representation(states)+','+str(action))
                print(str(acc)+'--MEMORY')

            remember(states, action, acc, prev_acc, new_state)
            prev_acc = acc
            states = new_state
            logger.update(states, acc, search_epochs)

        logger.log(states, acc, search_epochs, epsilon)
        replay(ql, min(len(memory), 256))


def update_state(state, action):
    '''
        Update the state, based on the action taken
    '''
    new_state = state.copy()
    # If we added a layer
    if actions[action] != 0:
        index = 0
        # Find the network's depth
        for depth in range(max_depth+1):
            start = depth*(len(actions)-1)
            actives = 0
            for i in range(len(actions)-1):
                actives += state[start+i]
            if actives == 0:
                index = start
                break
        # Set the next layer's state
        onehot_action = onehot_actions[action-1]
        for i in range(len(actions)-1):
            new_state[index+i] = onehot_action[i]
    return new_state


def clear_cache():
    '''
        Clear the model cahce file
    '''
    import os
    folder = 'weights_cache/'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def evaluate_q(ql):
    '''
        Create the best learnt network and evaluate it
    '''
    (x_train, y_train), (x_test, y_test) = get_data()
    states = np.zeros(states_no, dtype='int')
    model = None
    model_str = ''
    for x in range(max_depth):
        action = int(np.argmax(
            ql.predict(states.reshape(1, states_no))[0]
        ))
        model = expand_model(model, action)
        model_str += str(actions[action])+'x'
        states = update_state(states, action)
        print(states)
    print(model_str)

    acc = evaluate_model(model, x_train, y_train, x_test,
                         y_test, epochs=20, verbose_in=1)
    print(acc)


def zero_model(ql):
    '''
        Make model have close to zero output
    '''
    states = np.zeros(states_no, dtype='int')
    y_train = []
    x_train = []
    samples = 10000
    for i in range(samples):
        x_train.append(states.copy())
        y_train.append(np.zeros(len(actions), dtype='int'))
        states[np.random.randint(states_no)] = np.abs(
            states[np.random.randint(states_no)]-1)
    x_train = np.array(x_train).reshape(samples, states_no)
    y_train = np.array(y_train).reshape(samples, len(actions))
    ql.fit(x_train, y_train, epochs=5, verbose=1)


def update_q_target(ql, qv):
    '''
        Copy weights from learner network to target value network
    '''
    ql.save_weights('ql_weights')
    qv.load_weights('ql_weights')


def get_q_net():
    '''
        Create the Q network
    '''
    model = Sequential()
    model.add(Dense(128, activation='linear', input_dim=states_no))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.1))
    model.add(Dense(len(actions), activation='tanh'))
    model.compile(loss='mean_squared_error', optimizer='RMSProp')
    model.summary()
    return model


def main():
    global qv
    np.random.seed(7)


    clear_cache()

    ql = get_q_net()
    qv = get_q_net()

    zero_model(ql)
    update_q_target(ql, qv)

    train_q(ql)
    print('Done')
    evaluate_q(ql)

if __name__ == "__main__":
    main()
