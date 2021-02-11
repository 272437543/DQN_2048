import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten
from keras.activations import sigmoid, relu, softmax
from keras.optimizers import RMSprop
import numpy as np

class DeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_features,
                 lr=0.01,
                 reward_decay=0.9,
                 epsilon=0.96,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 train_epochs=10,
                 epsilon_increment=None,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.gamma = reward_decay
        self.epsilon_max = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = epsilon_increment
        self.epsilon = 0 if epsilon_increment is not None else self.epsilon_max
        self.train_epochs = train_epochs

        self.learn_step_counter = 0
        self._build_net_cnn()
        # 存储空间下标：0-31:S, 32-63:S_, 64:action, 65:reward
        self.memory = np.zeros((self.memory_size, self.n_features[0] * self.n_features[1] * 2 + 2))

    def preprocess_state(self, state):  # 预处理
        return np.log(state + 1) / 16

    def choose_action(self, state):
        state = state[np.newaxis, :, :, np.newaxis]
        state = self.preprocess_state(state)
        if np.random.uniform() < self.epsilon:
            action_value = self.q_eval_model.predict(state)
            action_index = np.argmax(action_value)
        else:
            action_index = np.random.randint(0, self.n_actions)
        return action_index

    def _build_net(self):
        self.q_eval_model = Sequential(name='evaluate net')
        self.q_eval_model.add(Dense(input_shape=[self.n_features], units=32, activation='relu'))
        self.q_eval_model.add(Dense(self.n_actions))

        self.q_target_model = Sequential(name='target net')
        self.q_target_model.add(Dense(input_shape=[self.n_features], units=32, activation='relu'))
        self.q_target_model.add(Dense(self.n_actions))
        rmsprop = RMSprop(lr=self.lr)
        self.q_eval_model.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])
        self.q_target_model.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])

    def _build_net_cnn(self):
        self.q_eval_model = Sequential(name='evaluate net')
        self.q_eval_model.add(Conv2D(16, (3, 3), input_shape=self.n_features, activation='relu', padding='SAME'))
        self.q_eval_model.add(Conv2D(32, (3, 3), activation='relu', padding='SAME'))
        self.q_eval_model.add(Flatten())
        self.q_eval_model.add(Dense(64, activation='relu'))
        self.q_eval_model.add(Dense(4, activation='softmax'))

        self.q_target_model = Sequential(name='target net')
        self.q_target_model.add(Conv2D(16, (3, 3), input_shape=self.n_features, activation='relu', padding='SAME'))
        self.q_target_model.add(Conv2D(32, (3, 3), activation='relu', padding='SAME'))
        self.q_target_model.add(Flatten())
        self.q_target_model.add(Dense(64, activation='relu'))
        self.q_target_model.add(Dense(4, activation='softmax'))
        self.q_target_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        self.q_eval_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def target_replace_op(self):
        p1 = self.q_eval_model.get_weights()
        self.q_target_model.set_weights(p1)

    def store_memory(self, s, s_, a, r):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        s = self.preprocess_state(s)
        s_ = self.preprocess_state(s_)
        memory = np.hstack((s, s_, [a, r]))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = memory
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_replace_op()
            print('target_params_replaced!')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        n_features = self.n_features[0] * self.n_features[1]
        s = batch_memory[:, 0:n_features].reshape([-1, self.n_features[0], self.n_features[1], 1])
        s_ = batch_memory[:, n_features:n_features*2].reshape([-1, self.n_features[0], self.n_features[1], 1])
        a = batch_memory[:, n_features*2].astype(np.int32)
        r = batch_memory[:, n_features*2+1]

        q_next = self.q_target_model.predict(s_)
        q_eval = self.q_eval_model.predict(s)

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, a] = r + self.gamma * np.max(q_next, axis=1)
        self.q_eval_model.fit(s, q_target, epochs=self.train_epochs, verbose=0)

        self.learn_step_counter += 1

    def save_model(self):
        self.q_eval_model.save('dqn2048_cnn.h5')
