import random
import sys

__author__ = 'colinc'

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from mpl_toolkits.axes_grid1 import ImageGrid

import get_data

MNIST_SIZE = (28, 28)


class FileHandler:
    """ A class that will read in a data filename, and hold onto training examples
    """

    def __init__(self, filename, n_to_load=None):
        self.filename = filename
        self.digits = self.__read_file(n_to_load)
        self._rf_clf = None

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

    @property
    def features(self):
        return np.array([pixel.features for pixel in self.digits])

    @property
    def labels(self):
        return np.array([pixel.label for pixel in self.digits])

    @property
    def rf_clf(self):
        """ Trains and returns a random forest classifier
        """
        if not self._rf_clf:
            self._rf_clf = RandomForestClassifier(n_estimators=100, oob_score=True)
            self._rf_clf = self._rf_clf.fit(normalize(self.features), self.labels)
        return self._rf_clf

    def show_bad_classifiers(self):
        """Displays a grid of up to 16 incorrectly classified examples
        """
        print("Model OOB score: {:.2f}%".format(100 * self.rf_clf.oob_score_))
        predictions = self.rf_clf.predict(self.features)
        misclassified = [self.digits[j] for j in range(len(self.digits)) if predictions[j] != self.labels[j]]
        print("Misclassified {:d} examples".format(len(misclassified)))
        misclassified = random.sample(misclassified, min(16, len(misclassified)))
        fig = plt.figure(1, (4., 4.))
        grid = ImageGrid(fig, 111, nrows_ncols=(4, 4),
                         axes_pad=0.1)
        for j, pixel in enumerate(misclassified):
            grid[j].imshow(pixel.data, cmap=plt.cm.binary)
            print("({:d}, {:d}: Predicted {:d}, actually {:d}".format(
                j // 4 + 1,
                j % 4 + 1,
                self.rf_clf.predict(pixel.features)[0],
                pixel.label))

        plt.show()


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

    @property
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
    """Returns a trained classifier
    """
    data = get_data.get_data_files()
    files = FileHandler(data['train'])
    return files.rf_clf


def __main():
    data = get_data.get_data_files()
    files = FileHandler(data['train'])
    files.show_bad_classifiers()


if __name__ == '__main__':
    sys.exit(__main())
