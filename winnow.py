__author__ = 'James Castiglione'

import numpy as np
from experience import Experience
import sys
import random


class Winnow:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.weights = np.ones(train_data[0].n)     # initialize weights to 1
        self.model = []

    def train_model(self):
        for ex in self.train_data:
            prediction = 0
            if np.dot(self.weights, ex.x) > ex.n / 2:
                prediction = 1

            if prediction == 0 and ex.y == 1:       # Promotion
                for i in xrange(ex.n):
                    if ex.x[i] == 1:
                        self.weights[i] *= 2

            if prediction == 1 and ex.y == 0:       # Elimination
                for i in xrange(ex.n):
                    if ex.x[i] == 1:
                        self.weights[i] = 0

        for i in xrange(len(self.weights)):
            if self.weights[i] != 0:
                self.model.append(i)

    def test_model(self):
        n = 0
        correct = 0
        for test in self.test_data:
            if self.predict(test.x) == test.y:
                correct += 1
            n += 1
        return correct, n

    def predict(self, x):
        prediction = 0
        for index in self.model:
            if x[index] == 1:
                prediction = 1
                break
        return prediction

    def print_weights(self):
        result = ''
        for w in self.weights:
            result += str(w) + ' '
        print result

    def randomize_data(self):
        random.shuffle(self.train_data)
        random.shuffle(self.test_data)


def main(argv):
    training_file = argv[0]
    test_file = argv[1]

    w = Winnow(read_data_winnow(training_file), read_data_winnow(test_file))
    print w.train_data[0]
    w.randomize_data()
    w.train_model()
    w.print_weights()
    print w.model
    print w.test_model()


def parse_config_winnow(filename):
    ''' Takes the config filename and returns a dict with category : 0|1 '''

    config_map = {'y': 1, 'n': 0}

    with open(filename) as f:
        labels = f.readline().strip('\n').split(',')
        config_map[labels[0]] = 1
        config_map[labels[1]] = 0

    return config_map


def parse_data_winnow(lines, config_map):
    data = []
    for line in lines:
        values = line.strip('\n').split(',')
        x = np.array([config_map[feat] for feat in values[2:]]) # Maps features to
        e = Experience(values[0], config_map[values[1]], x)					# 0|1 value
        data.append(e)
    return data


def read_data_winnow(filename):
    ''' Reads data file and returns a list of Experience objects '''
    config_map = parse_config_winnow(filename.replace('.data', '.config'))

    with open(filename) as f:
        lines = f.readlines()
    return parse_data_winnow(lines, config_map)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))