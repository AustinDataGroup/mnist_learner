import random
import sys
import get_data

__author__ = 'colinc'

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.axes_grid1 import ImageGrid

MNIST_SIZE = (28, 28)


class FileHandler:
    """ A class that will read in a data filename, and hold onto training examples
    """

    def __init__(self, train_filename, test_filename, n_to_load=None):
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.train_digits = self.__read_file(self.train_filename, n_to_load)
        self.test_digits = self.__read_file(self.test_filename)
        self._rf_clf = None
        self._lr_clf = None

    @staticmethod
    def __read_file(filename, n_lines=None):
        all_data = []
        has_labels = False
        with open(filename) as buff:
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
        row_num = np.random.randint(0, len(self.train_digits))
        self.train_digits[row_num].show()

    def features(self, data_set='train'):
        if data_set == 'train':
            return normalize(np.array([pixel.features for pixel in self.train_digits]))
        return normalize(np.array([pixel.features for pixel in self.test_digits]))

    @property
    def labels(self):
        return np.array([pixel.label for pixel in self.train_digits])

    @property
    def lr_clf(self):
        """ Trains and returns a random forest classifier
        """
        if not self._lr_clf:
            self._lr_clf = LogisticRegression()
            self._lr_clf = self._lr_clf.fit(self.features(), self.labels)
        return self._lr_clf

    @property
    def rf_clf(self):
        """ Trains and returns a random forest classifier
        """
        if not self._rf_clf:
            self._rf_clf = RandomForestClassifier(n_estimators=10, oob_score=True)
            self._rf_clf = self._rf_clf.fit(self.features(), self.labels)
        return self._rf_clf

    def show_bad_classifiers(self):
        """Displays a grid of up to 16 incorrectly classified examples
        """
        print("Model OOB score: {:.2f}%".format(100 * self.rf_clf.oob_score_))
        predictions = np.argmax(self.rf_clf.oob_decision_function_, 1)

        for j in range(predictions.shape[0]):
            self.train_digits[j].prediction = predictions[j]
            self.train_digits[j].decision_function = self.rf_clf.oob_decision_function_[j, :]

        misclassified = [digit for digit in self.train_digits if digit.prediction != digit.label]
        print("Misclassified {:d} examples".format(len(misclassified)))
        misclassified = random.sample(misclassified, min(16, len(misclassified)))
        fig = plt.figure(1, (4., 4.))
        grid = ImageGrid(fig, 111, nrows_ncols=(4, 4),
                         axes_pad=0.1)
        for j, pixel in enumerate(misclassified):
            grid[j].imshow(pixel.data, cmap=plt.cm.binary)
            print("({:d}, {:d}): Predicted {:d}, actually {:d}".format(
                j // 4 + 1,
                j % 4 + 1,
                pixel.prediction,
                pixel.label))

        plt.show()

    def write_predictions(self, write_filename='mnist/data/predictions.csv'):
        """ Loads the test data set and writes predictions in the manner
        directed by http://www.kaggle.com/c/digit-recognizer/data
        """
        predictions = list(self.rf_clf.predict(self.features('test')))
        with open(write_filename, 'w') as buff:
            buff.write("ImageId,Label\n")
            for j, prediction in enumerate(predictions):
                buff.write("{:d},{:d}\n".format(j + 1, prediction))


class Digit:
    def __init__(self, data, digit_label=None):
        self.data = data
        self.label = digit_label
        self.prediction = "None"
        self.decision_function = None

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
    data = get_data.get_data_files('mnist')
    files = FileHandler(data['train'], data['test'])
    return files.rf_clf


def __main():
    data = get_data.get_data_files('mnist')
    files = FileHandler(data['train'], data['test'])
    print(files.rf_clf)
    files.write_predictions()


if __name__ == '__main__':
    sys.exit(__main())
