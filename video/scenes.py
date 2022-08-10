from manim import *
from video.code_explained import CodeExplained, line_centerer
import os

os.environ["PATH"] = os.environ["PATH"] + r";C:\Users\Heidi\Desktop\epq\FFMPEG\bin\\"


class Intro(Scene):
    def construct(self):
        text = Text("Deep Q Learning in Python", font_size=42, color=WHITE)
        text2 = Text("Deep Q Learning Series - Part 2", font_size=36, color=ORANGE)
        text.set_y(0.25)
        text2.set_y(-0.3)
        self.play(Write(text), run_time=3)
        self.play(Write(text2), run_time=2)
        self.wait(1)
        self.play(Unwrite(text), Unwrite(text2), run_time=1)

        self.wait(0.8)

        dependencies = Text("Dependencies")
        dependencies.set_x(-4)
        dependencies.set_y(3)

        self.play(Write(dependencies))

        dep_list = Text(
            "• numpy\n• Neural Network in description\n\t\tor other NN library",
            line_spacing=1
        )

        self.wait(0.5)

        self.play(
            Write(dep_list)
        )
        self.wait(3)
        self.play(
            Unwrite(dep_list),
            Unwrite(dependencies)
        )
        self.wait(1)
        nn_warning = line_centerer(
            "This tutorial will use my own Neural Network in the description. Methods can easily be adapted to use others, such as keras. Methods to change are marked with *.",
            font_size=36)

        self.play(Write(nn_warning))
        self.wait(6)
        self.play(Unwrite(nn_warning))

code_snippets = [
    {
        "code":
            """
# imports from custom neural network in description
from neural_network_classes import NeuralNetwork, ConnectedLayer
from activation_functions import relu, linear

import numpy as np
import random
            """,
        "explained": "Import requirements. Custom neural network can be replaced with other libraries e.g. tensorflow. *"
    },
    {
        "code":
            """
class QLearning:
    def __init__(self, state_n, actions_n):
        self.state_n = state_n
        self.actions_n = actions_n

        # hyperparameters
        self.learning_rate = 0.00001
        self.discount_rate = 0.95
        self.exploration_probability = 1
        self.exploration_decay = 0.0001
        
        # to be implemented
        self.network = self.create_network()

        # a list of experiences per episode
        self.experience_buffer = []
        # fraction of experiences to train on            
        self.train_amount = 0.7
            """,
        "explained": "Initial attributes such as hyperparameters, and creating the network."
    },
    {
        "code":
            """
def create_network(self):
    return NeuralNetwork(
        [
            ConnectedLayer(relu, self.state_n, 12),
            ConnectedLayer(relu, 12, 18),
            ConnectedLayer(relu, 18, 12),
            ConnectedLayer(linear, 12, self.actions_n)
        ], learning_rate=self.learning_rate)
            """,
        "explained": "Using a neural network of your choice, create the deep NN. *"
    },
    {
        "code":
            """
def decay_exploration_probability(self):
    # Decrease exploration exponentially
    # new = old * e^-decay
    self.exploration_probability = self.exploration_probability * np.exp(-self.exploration_decay)
            """,
        "explained": "Decay epsilon (epsilon greedy policy)"
    },
    {
        "code":
            """
def get_q_values(self, state):
    return self.network.predict(state).tolist()
            """,
        "explained": "Uses the neural network to make Q value predictions. *"
    },
    {
        "code":
            """
def get_action(self, state):
    # get action for state (largest q value)
    # if probability is correct, choose random action (epsilon greedy)
    if random.random() < self.exploration_probability:
        action = random.randint(0, self.actions_n-1)
        return action
    q_values = self.get_q_values(state)
    return q_values.index(max(q_values))
            """,
        "explained": "Method to get action for the agent in a given state. Includes epsilon greedy policy."
    },
    {
        "code":
            """
def update_experience_buffer(self, state, action, reward):
    self.experience_buffer.append((tuple(state), action, reward))

    if len(self.experience_buffer) > 5000:
        self.experience_buffer.pop(0)
            """,
        "explained": "Add a new experience to the experience buffer. Maximum of 5000."
    },
    {
        "code":
            """
def fit(self, state, correct_q_values):
    self.network.train([state], [correct_q_values], epochs=1, log=False)
            """,
        "explained": "Backpropogate the error of the neural network. *"
    },
    {
        "code":
            """
def train(self):
    if not len(self.experience_buffer):
        return
            """,
        "explained": "Start of train method. Return if no experiences are available."
    },
    {
        "code":
            """
def train(self):
    ...
    # number of experiences to train on
    training_experiences_count = int(len(self.experience_buffer) * self.train_amount)
    # sample indices (in experience buffer) of experiences to train on, randomly
    experiences_indices = random.sample(range(len(self.experience_buffer)), training_experiences_count)
            """,
        "explained": "Create a random list of the index of experiences to train on."
    },
    {
        "code":
        """
def train (self):
    ...
    for experience_num in experiences_indices:
        # experience: (state, action, reward)
        experience = self.experience_buffer[experience_num]
    
        state = experience[0]
        action = experience[1]
        reward = experience[2]
        """,
        "explained": "Retrieve data for this experience."
    },
    {
        "code":
            """
def train(self):
    ...
    for experience_num in experiences_indices:
        ...
        try:
            next_experience = self.experience_buffer[experience_num + 1]
            max_next_q_value = max(self.get_q_values(next_experience[0]))
        except IndexError:
            max_next_q_value = 0
            """,
        "explained": "Find Max Next Q Value, if available, or set to 0."
    },
    {
        "code":
            """
def train(self):
    ...
    for experience_num in experiences_indices:
        ...
        # bellman optimality equation
        q_target = reward + (self.discount_rate * max_next_q_value)
            """,
        "explained": "Find Q target using Bellman Optimality Equation."
    },
    {
        "code":
            """
def train(self):
    ...
    for experience_num in experiences_indices:
        ...
        # if predicted Q values were (0.1, 0.2, 0.3)
        # and action 1 was taken
        # correct q values are (0.1, q_target, 0.3)
        
        correct_q_values = self.get_q_values(state)
        correct_q_values[action] = q_target
    
        self.fit(state, correct_q_values)
            """,
        "explained": "Finally, find correct Q values and fit to neural network."
    },
    {
        "code":
            r"""
def train(self):
    ...
    print("Exploration:", self.exploration_probability)
    # print total of rewards for episode
    print("Reward:", sum(map(lambda x: x[2], self.experience_buffer)))
    print("============================================\n")

    self.experience_buffer = []
            """,
        "explained": "Print some stats and reset the buffer."
    }
]


class Code(Scene):
    def construct(self):
        for code in code_snippets:
            screen_object = CodeExplained(
                scene=self,
                **code
            )

            screen_object.play_anims(True)
            screen_object.play_anims(False)

            self.wait(1.3)


class Outro(Scene):
    def construct(self):
        desc = Text("Full code in description.")
        desc.set_y(1)

        like = Text("Leave a like if you enjoyed!", color=ORANGE)
        like.set_y(-1)
        self.play(Write(desc), Write(like))
        self.wait(7)
        self.play(Unwrite(desc), Unwrite(like))


class All(Scene):
    def construct(self):
        Intro.construct(self)
        Code.construct(self)
        Outro.construct(self)


if __name__ == "__main__":
    os.system("pipenv run manim scenes.py -qp")
