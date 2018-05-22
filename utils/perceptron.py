"""Simple perceptron."""

from random import shuffle

import matplotlib.pyplot as plt
import numpy as np

from utils.helpers import progress, debug

class Perceptron():
    """Simple perceptron implementation without bias."""

    def __init__(self, eta=0.01, epochs=50, avg=False, shuffle=False):
        """Setup perceptron instance."""
        self.eta = eta
        self.epochs = epochs
        self.avg = avg
        self.shuffle = shuffle
        self.converged = False

    def train(self, X):
        """Given a bag-of-words matrix, train the perceptron."""
        lv = X[0][0].shape[0]
        self.errors = []

        self.w = np.zeros((self.epochs + 2, lv))
        self.w[0] = np.random.rand(lv)
        self.i = 0

        for i in range(1, self.epochs + 1):
            progress('--- perceptron: epoch', i)

            if self.shuffle:
                shuffle(X)

            self.w[i] = self.w[i - 1]

            error = 0

            for x, y, z in X:
                fx = self.predict(x, i)
                debug('--- perceptron:', 'y:', y, 'fx', fx)

                if y != fx:
                    self.w[i] = self.w[i] + self.eta * x * y
                    error += np.abs(y - fx)

            self.errors.append(error)
            self.i = i

            if error == 0:
                self.converged = True
                break

        self.i = self.w.shape[0] - 1
        conv_epochs = self.convergance_epochs()

        if self.avg:
            self.w[self.i] = np.mean(self.w[:conv_epochs], axis=0)
        else:
            self.w[self.i] = self.w[conv_epochs]

        return self

    def feed(self, x, i=None):
        """Calculate score of x using the given or latest w."""
        if i is None:
            i = self.i

        debug('--- perceptron: dot:', np.dot(self.w[i], x))
        return np.dot(self.w[i], x)

    def predict(self, x, i=None):
        """Predict class of x using the given or latest w."""
        if i is None:
            i = self.i

        return int(np.sign(self.feed(x, i)))

    def plot_training_error(self):
        """Show plot of total errors per epoch."""
        fig, ax = plt.subplots()
        ax.plot(list(range(1, self.convergance_epochs() + 1)), self.errors)
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.show()

    def convergance_epochs(self):
        """Get the number of epochs taken to achieve 0 error."""
        return len(self.errors)

    def final_w(self):
        """Get the last w."""
        return self.w[self.i]
