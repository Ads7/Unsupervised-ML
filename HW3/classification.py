from sklearn import linear_model
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from HW2 import DATA_DIR


class Classify(object):
    iterations = 6
    top = 30

    def __init__(self, X, Y, x_test, y_test):
        self.X = X
        self.Y = Y
        self.x_test = x_test
        self.y_test = y_test

    def logistic_regression(self):
        logistic = linear_model.LogisticRegression()
        logistic.fit(self.X, self.Y)
        labels = logistic.predict(self.x_test)
        print logistic.coef_
        return labels

    def cal_purity(self, prediction, labels):
        """
        :return: : returns the purity and the gini coeff
        :rtype: float,float
        """
        # cal_purity(gmm.predict(FMNIST[:, :-1]), 2, FMNIST[:, [-1]], FMNIST[:, [-1]].shape[0], True)
        purity = 0.0
        K = np.unique(prediction).shape[0]
        print "clusters " + str(K)
        SIZE = labels.shape[0]
        for i in range(K):
            indexes = np.argwhere(prediction == i)
            counts = np.bincount(labels[indexes.transpose()[0]])
            purity += counts.max()
        purity = purity / SIZE
        print "purity " + str(purity)
        return purity

    def start(self):
        for i in range(self.iterations):
            predictions = self.logistic_regression()
            self.cal_purity(predictions, self.y_test)


if __name__ == '__main__':
    MNIST = input_data.read_data_sets(DATA_DIR + "MNIST_data/")
    c = Classify(MNIST.train.images, MNIST.train.labels, MNIST.test.images, MNIST.test.labels)
    c.start()
