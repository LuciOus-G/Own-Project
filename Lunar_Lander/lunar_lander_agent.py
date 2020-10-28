from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
import numpy as np

# this the Deep Q-Network which this will gave a new data every frame
class DQN:
    def __init__(self, max_size, in_shape, n_act, dicrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self. in_shape = in_shape
        self.discrete = dicrete
        self.st_memory = np.zeros((self.mem_size, in_shape))
        self.st_memory_new = np.zeros((self.mem_size, in_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.act_memory = np.zeros((self.mem_size, n_act), dtype=dtype)
        self.rew_memory = np.zeros(self.mem_size)
        self.term_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.st_memory[index] = state
        self.st_memory_new[index] = state_
        self.rew_memory[index] = reward
        self.term_memory[index] = 1 - int(done)

        if self.discrete:
            actions = np.zeros(self.act_memory.shape[1])
            actions[action] = 1.0
            self.act_memory[index] = actions
        else:
            self.act_memory[index] = action

        self.mem_cntr += 1

    def buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.st_memory[batch]
        states_ = self.st_memory_new[batch]
        reward = self.rew_memory[batch]
        actions = self.act_memory[batch]
        terminal = self.term_memory[batch]

        return states, actions, reward, states_, terminal

# the model is actualy simple
def DQN_model(lr, n_act, input_dims, fc1, fc2):
    model = Sequential()
    model.add(Dense(fc1, input_shape=(input_dims, )))
    model.add(Activation('relu'))
    model.add(Dense(fc2))
    model.add(Activation('relu'))
    model.add(Dense(n_act))
    model.compile(optimizer=Adam(lr=lr), loss='mse', metrics=['accuracy'] )

    return model

# and Agent is the program that operate the enviroment
class Agent(object):
    def __init__(self, alpha, gamma, n_act, epsilon, batch_size, in_dims, epsilon_dec=0.996,
                 epsilon_end=0.01, mem_sizes=1000000, fname='dqn-model.h5'):

        self.act_space = [i for i in range(n_act)]
        self.n_actions = n_act
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_end
        self.epsilon_dec = epsilon_dec
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = DQN(mem_sizes, in_dims, n_act, dicrete=True)
        self.evaluate = DQN_model(alpha,n_act, in_dims, 256, 256)

    def data_memory(self, state, act, reward, state_, done):
        self.memory.store(state, act, reward, state_, done)

    def decision(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()

        if rand < self.epsilon:
            action = np.random.choice(self.act_space)
        else:
            actions = self.evaluate.predict(state)
            action = np.argmax(actions)

        return action
    # this thing that make the agent optimal every frame
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state , done = self.memory.buffer(self.batch_size)

        action_value = np.array(self.act_space, dtype=np.int8)
        action_indices = np.dot(action, action_value)

        evaluate_q = self.evaluate.predict(state)
        next_q = self.evaluate.predict(new_state)

        target_q = evaluate_q.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        target_q [batch_index, action_indices] = reward + self.gamma * np.max(next_q, axis=1) * done

        _ = self.evaluate.fit(state, target_q, verbose=0)

        self.epsilon = (self.epsilon*self.epsilon_dec) if self.epsilon > self.epsilon_min else self.epsilon_min


    def save_model(self):
        self.evaluate.save(self.model_file)

