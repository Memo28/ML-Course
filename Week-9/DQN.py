import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 #discount rate
        self.epsilon = 1.0 #ecploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        #NN for Deep-Q learning

        model = Sequential()
        model.add(Dense(24, input_dim = self.state_size, activation ='relu'))
        model.add(Dense(24, activation = 'relu'))
        model.add(Dense(self.action_size, activation = 'linear'))
        model.compile(loss = 'mse', optimizer=Adam(lr = self.learning_rate))

        return model

    #Save NN states in memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    #Initialize the NN with a random action for learning purposes
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                #Matematical respresentation Q-learning
                target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs = 1, verbose = 0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



#Main

if __name__ == "__main__":
    #Initializing the environment

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-ddqn.h5")
    done = False
    batch_size = 32


    #Itarete the game
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        #Trying to keep the pole for 500
        for time in range(500):
            #Display the environment
            env.render()

            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])

            #Saving the last state
            agent.remember(state, action, reward, next_state, done)

            #Replacing state with next state
            state = next_state

            if done:
                print("episodes: {}/{}, score : {}".format(e, EPISODES, time))

                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            #Retraining the agent
            #agent.replay(32)


