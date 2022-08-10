# imports from custom neural network in description
from neural_network_classes import NeuralNetwork, ConnectedLayer
from activation_functions import relu, linear

import numpy as np
import random


class QLearning:
    def __init__(self, state_n, actions_n):
        self.state_n = state_n
        self.actions_n = actions_n

        # hyperparameters copied from global settings
        self.learning_rate = 0.00001
        self.discount_rate = 0.95
        self.exploration_probability = 1
        self.exploration_decay = 0.0001

        self.network = self.create_network()

        # a list of experiences per episode
        self.experience_buffer = []
        self.train_amount = 0.7  # fraction of experiences to train on

    def create_network(self):
        """
        Uses Custom model as Neural Network in Deep Q Learning
        """
        return NeuralNetwork(
            [
                ConnectedLayer(relu, self.state_n, 12),
                ConnectedLayer(relu, 12, 18),
                ConnectedLayer(relu, 18, 12),
                ConnectedLayer(linear, 12, self.actions_n)
            ], learning_rate=self.learning_rate)

    def decay_exploration_probability(self):
        # Decrease exploration exponentially
        # y = e^(-decay*x)
        # so
        # new = old * e^-decay
        self.exploration_probability = self.exploration_probability * np.exp(-self.exploration_decay)

    def get_action(self, state):
        # get action for state (largest q value)
        # if probability is correct, choose random action (epsilon greedy)
        if random.random() < self.exploration_probability:
            action = random.randint(0, self.actions_n-1)
            return action

        q_values = self.get_q_values(state)
        return q_values.index(max(q_values))

    def get_q_values(self, state):
        return self.network.predict(state).tolist()

    def update_experience_buffer(self, state, action, reward):
        self.experience_buffer.append((tuple(state), action, reward))

        if len(self.experience_buffer) > 5000:
            self.experience_buffer.pop(0)

    def fit(self, state, correct_q_values):
        self.network.train([state], [correct_q_values], epochs=1, log=False)

    def train(self):
        if not len(self.experience_buffer):
            return

        print("============================================")
        # number of experiences to train on
        training_experiences_count = int(len(self.experience_buffer) * self.train_amount)
        # sample indicies (in experience buffer) of experiences to train on, randomly
        experiences_indices = random.sample(range(len(self.experience_buffer)), training_experiences_count)

        for experience_num in experiences_indices:
            # experience: (state, action, reward)
            experience = self.experience_buffer[experience_num]

            state = experience[0]
            action = experience[1]
            reward = experience[2]

            try:
                next_experience = self.experience_buffer[experience_num + 1]
                max_next_q_value = max(self.get_q_values(next_experience[0]))
            except IndexError:
                max_next_q_value = 0

            # bellman optimality equation
            q_target = reward + (self.discount_rate * max_next_q_value)

            correct_q_values = self.get_q_values(state)

            # if predicted Q values were (0.1, 0.2, 0.3)
            # and action [1] was taken
            # correct q values are (0.1, q_target, 0.3)

            correct_q_values[action] = q_target

            self.fit(state, correct_q_values)

        print("Exploration:", self.exploration_probability)
        # print total of rewards for episode
        print("Reward:", sum(map(lambda x: x[2], self.experience_buffer)))
        print("============================================\n")

        self.experience_buffer = []
