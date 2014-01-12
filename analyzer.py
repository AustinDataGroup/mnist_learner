import sys

__author__ = 'colinc'

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import get_data

MNIST_SIZE = (28, 28)


class FileHandler:
    """ A class that will read in a data filename, and hold onto training examples
    """

    def __init__(self, filename, n_to_load=None):
        self.filename = filename
        self.digits = self.__read_file(n_to_load)

    def __read_file(self, n_lines=None):
        all_data = []
        has_labels = False
        with open(self.filename) as buff:
            row = buff.next().split(',')
            if len(row) != MNIST_SIZE[0] * MNIST_SIZE[1]:
                has_labels = True
            for row_num, row in enumerate(buff):
                if n_lines and row_num >= n_lines:
                    break
                label, data = None, map(int, row.split(','))
                if has_labels:
                    label = data.pop(0)
                all_data.append(Digit(np.array(data).reshape(MNIST_SIZE), label))
        return all_data

    def print_random_example(self):
        row_num = np.random.randint(0, len(self.digits))
        self.digits[row_num].show()

    def features(self):
        return np.array([pixel.features() for pixel in self.digits])

    def labels(self):
        return np.array([pixel.label for pixel in self.digits])

class Digit:
    def __init__(self, data, digit_label=None):
        self.data = data
        self.label = digit_label

    def __repr__(self):
        if self.label:
            return "Labelled digit {:d}".format(self.label)
        return "Unlabelled {:d}x{:d} digit".format(*self.data.shape)

    def show(self):
        print(self.__repr__())
        plt.imshow(self.data, cmap=plt.cm.binary)
        plt.show()

    def ink(self):
        """ Prints the average value of all the pixels -- intuitively, a "1" might use less
        ink than an "8".
        """
        return self.data.mean()

    def features(self):
        """ Gathers up a feature set for the digits and delivers a single numpy array
        """
        return np.array(list(self.data.flatten()) + [self.ink()])


def normalize(matrix):
    """ Normalizes a matrix, subtracting the mean of each column and dividing by the range
    of values
    """
    return (matrix - matrix.mean(0)) / (matrix.max() - matrix.min())


def train_model():
    data = get_data.get_data_files()
    files = FileHandler(data['train'])
    clf = RandomForestClassifier(n_estimators=100, oob_score=True)
    clf = clf.fit(normalize(files.features()), files.labels())
    return clf


def __main():
    data = get_data.get_data_files()
    files = FileHandler(data['train'], 20)
    print normalize(files.labels())


if __name__ == '__main__':
    sys.exit(__main())
