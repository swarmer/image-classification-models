#!/usr/bin/env python
from collections import deque
import math
import statistics

import numpy as np
import scipy.special


class NeuralNetwork:
    def __init__(
            self, input_count=2, hidden_count=300, output_count=3,
            learning_rate=0.3,
    ):
        self.input_count = input_count
        self.hidden_count = hidden_count
        self.output_count = output_count
        self.lr = learning_rate

        self.dropout_threshold = 1000000
        self.dropout_steps = 5000
        self.dropout_rate = 1.0

        self.whi, self.woh = self._initial_weights()

    def _initial_weights(self):
        mean = 0.0
        stddev = 1 / math.sqrt(self.hidden_count)

        return (
            np.random.normal(mean, stddev, (self.hidden_count, self.input_count)),
            np.random.normal(mean, stddev, (self.output_count, self.hidden_count)),
        )

    def loss(self, outputs, targets):
        return (outputs - targets) ** 2

    def loss_derivative(self, outputs, targets):
        return outputs - targets

    def activation(self, x):
        return scipy.special.expit(x)

    def train_pass(self, inputs, targets):
        inputs = inputs.reshape(self.input_count)

        hidden_values = self._query_hidden(inputs)
        output_values = self._query_outputs(hidden_values)

        ru = self.loss_derivative(output_values, targets) * output_values * (1.0 - output_values)
        dw = (
            self.lr *
            np.outer(
                -ru,
                hidden_values,
            )
        )
        self.woh += dw

        hidden_ru = self.woh.T @ ru * hidden_values * (1.0 - hidden_values)
        dw = (
            self.lr *
            np.outer(
                -hidden_ru,
                np.array(inputs),
            )
        )
        self.whi += dw

        return sum(self.loss(output_values, targets))

    def train(self, training_pairs):
        loss_interval = 1000
        losses = deque(maxlen=loss_interval)

        for i, (inputs, targets) in enumerate(training_pairs):
            if i >= self.dropout_threshold and i % self.dropout_steps == 0:
                self.lr *= self.dropout_rate

            loss = self.train_pass(inputs, targets)
            losses.append(loss)

            if i % loss_interval == 0:
                print(f'Running mean loss: {statistics.mean(losses)}')
                print(f'Learning rate: {self.lr:.5f}')

    def _query_hidden(self, inputs):
        hidden_sums = self.whi @ inputs
        hidden_values = self.activation(hidden_sums)
        return hidden_values

    def _query_outputs(self, hidden_values):
        output_sums = self.woh @ hidden_values
        output_values = self.activation(output_sums)

        return output_values

    def query(self, inputs):
        inputs = inputs.reshape(self.input_count)

        hidden_values = self._query_hidden(inputs)
        output_values = self._query_outputs(hidden_values)

        return output_values
