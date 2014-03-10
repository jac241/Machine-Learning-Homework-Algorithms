__author__ = 'James Castiglione'
import numpy as np
from experience import Experience
import sys
import random


class Perceptron:

    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.weights = np.zeros(train_data[0].n)     # initialize weights to 0

    def train_model(self):
        for ex in self.train_data:
            hx = 1 if np.dot(self.weights, ex.x) > 0 else -1    # returns 1 if + and -1 if -1
            for i in xrange(len(self.weights)):
                self.weights[i] += 0.05 * (ex.y - hx) * ex.x[i]

    def test_model(self):
        n = 0
        correct = 0
        for test in self.test_data:
            if self.predict(test.x) == test.y:
                correct += 1
            n += 1
        return correct, n

    def predict(self, x):
        return 1 if np.dot(self.weights, x) > 0 else -1

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
    p = Perceptron(read_data_perceptron(training_file), read_data_perceptron(test_file))
    print p.train_data[0]
    p.randomize_data()
    p.train_model()
    p.print_weights()
    #print p.model
    print p.test_model()


def read_data_perceptron(filename):
    ''' Reads data file and returns a list of Experience objects '''
    with open(filename) as f:
        lines = f.readlines()
    config_map = parse_config_perceptron(filename.replace('.data', '.config'))
    return parse_data_perceptron(lines, config_map)


def parse_data_perceptron(lines, config_map):    
    data = []
    for line in lines:
        values = line.strip('\n').split(',')
        x = [1]
        for feat in values[2:]:
            x.append(config_map[feat])
        x = np.array(x)   # Maps features to
        e = Experience(values[0], config_map[values[1]], x)					# 0|1 value
        data.append(e)
    return data


def parse_config_perceptron(filename):
    ''' Takes the config filename and returns a dict with category : 0|1 '''

    config_map = {'y': 1, 'n': 0}

    with open(filename) as f:
        labels = f.readline().strip('\n').split(',')
        config_map[labels[0]] = 1
        config_map[labels[1]] = -1

    return config_map

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
