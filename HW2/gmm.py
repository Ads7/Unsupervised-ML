import random

import numpy as np
from sklearn import mixture

from HW2 import DATA_DIR
from scipy import stats


class GMM(object):
    SIZE = None
    MAX_ITR = 100  # max iteration for stopping criteria
    responsibilities = None
    mu = None  # means
    sigma = None  # variance
    weights = []
    TOLERANCE = 1.e-4  # tolerance factor for stopping criteria
    X = None
    dimension = None
    K = 2
    p_clus = None

    def __init__(self, k=2, filename="2gaussian.txt"):
        """

        :param k: int number of gaussian mixture
        :param filename: filename for the data file
        """
        self.K = k
        self.X = self.file_to_vec(filename)
        self.SIZE, self.dimension = self.X.shape

        #  assigning equal weights to all
        self.weights = [1.0 / k] * (k - 1)
        self.weights.append(1.0 - sum(self.weights))

        self.mu = self.X[[random.randint(1, self.SIZE) for x in range(self.K)]]
        self.sigma = [np.identity(self.dimension) for i in range(k)]
        self.p_clus = np.zeros((self.SIZE, self.K))

    @staticmethod
    def file_to_vec(filename):
        """

        :type filename: string take filename and converts to numpy vectors
        """
        print("Loading data")
        vectors = np.genfromtxt(DATA_DIR + '/' + filename, delimiter=' ')
        print("Data loaded")
        return vectors

    def expectation(self):
        """
        Updates the membership probabilities
        """
        for j in range(self.K):
            self.p_clus[:, j] = self.weights[j] * np.apply_along_axis(
                lambda x: stats.multivariate_normal.pdf(x=x, cov=self.sigma[j], mean=self.mu[j]), 1, self.X)
        tmp = self.p_clus.sum(axis=1)
        tmp[tmp == 0] = 1
        self.p_clus = self.p_clus / tmp[:, None]

    def maximisation(self):
        """
        Updates the mean and weights factors
        """
        for i in range(self.K):
            p = self.p_clus[:, [i]]
            self.mu[i] = np.sum(p * self.X, axis=0) / np.sum(p)
            self.weights[i] = np.sum(p) / self.SIZE
            tmp = self.X - self.mu[i]
            self.sigma[i] = np.dot(np.transpose(tmp), np.multiply(tmp, p)) / np.sum(p)

    def start(self):
        """
        To initiate the training process
        """
        ex = 0
        while ex < self.MAX_ITR:
            old = self.p_clus.copy()
            self.expectation()
            self.maximisation()
            if np.allclose(self.p_clus, old, atol=self.TOLERANCE):
                break
            ex += 1
            print(ex)
        pos = np.bincount(np.argmax(self.p_clus, axis=1))
        print('iterations taken: ', ex)
        for i in range(self.K):
            print(self.mu[i])
            print(self.sigma[i])
            print(pos[i])


def cal_purity(distances, K, labels, SIZE, check=False):
    """
    :return: : returns the purity and the gini coeff
    :rtype: float,float
    """
    purity = 0.0
    for i in range(K):
        indexes = np.argwhere(distances == i)
        if check:
            counts = np.bincount(labels[indexes.transpose()[0]].transpose()[0].astype('int64'))
        else:
            counts = np.bincount(labels[indexes.transpose()[0]])
        purity += counts.max()
    purity = purity / SIZE
    return purity


def gmm_fashion():
    from tensorflow.examples.tutorials.mnist import input_data
    print("loading ...")
    data = input_data.read_data_sets(DATA_DIR + '/fashion',
                                     source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')

    print("data loaded")
    gmm = mixture.GaussianMixture(n_components=10, covariance_type='diag')
    gmm.fit(data.train.images)
    print("cov", gmm.covariances_)
    print("mean", gmm.means_)
    print("purity", cal_purity(gmm.predict(data.train.images), 10, data.train.labels, data.train.images.shape[0]))
    gmm.predict(data.train.images)
    print(np.mean(gmm.predict(data.train.images).ravel() == data.train.labels.ravel()))

    # running Spam emails
    print("running spam emails")
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='diag')
    FMNIST = np.genfromtxt(DATA_DIR + '/spambase.data', delimiter=',')
    gmm.fit(FMNIST[:, :-1])
    print("cov", gmm.covariances_)
    print("mean", gmm.means_)
    print("purity", cal_purity(gmm.predict(FMNIST[:, :-1]), 2, FMNIST[:, [-1]], FMNIST[:, [-1]].shape[0], True))
    print(np.mean(gmm.predict(FMNIST[:, :-1]).ravel() == FMNIST[:, [-1]].ravel()))


if __name__ == '__main__':
    print("2gaussian.txt")
    g = GMM()
    g.start()
    print("3gaussian.txt")
    g = GMM(k=3, filename="3gaussian.txt")
    g.start()
    print("Question 4")
    gmm_fashion()
