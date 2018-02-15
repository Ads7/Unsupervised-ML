# 2gaussian.txt
# 3gaussian.txt
# mean_1 [3,3]); cov_1 = [[1,0],[0,3]]; n1=2000 points
# mean_2 =[7,4]; cov_2 = [[1,0.5],[0.5,1]]; ; n2=4000 points
import random

import numpy as np
from sklearn import mixture

from HW2 import DATA_DIR
import seaborn as sns

# For plotting
import matplotlib.pyplot as plt
# for normalization + probability density function computation
from scipy import stats


class Gaussian(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, data):
        u = (data - self.mu) / abs(self.sigma)


class GMM(object):
    SIZE = None
    MAX_ITR = 100
    responsibilities = None
    mu = None  # means
    sigma = None  # variance
    weights = []
    X = None
    dimension = None
    K = 2
    p_clus = None

    def __init__(self, k=2, path="2gaussian.txt"):
        self.K = k
        self.X = self.file_to_vec(path)
        self.SIZE, self.dimension = self.X.shape

        #  assigning equal weights to all
        self.weights = [1.0 / k] * (k - 1)
        self.weights.append(1.0 - sum(self.weights))

        self.mu = self.X[[random.randint(1, self.SIZE) for x in range(self.K)]]
        self.sigma = [np.identity(self.dimension) for i in range(k)]
        self.p_clus = np.zeros((self.SIZE, self.K))

    def file_to_vec(self, path):
        print "Loading data"
        return np.genfromtxt(DATA_DIR + '/' + path, delimiter=' ')
        print "Data loaded"

    def pdf(self, sigma, mu, x):
        return (1.0 / np.sqrt(((2 * np.pi) ** self.dimension / 2) * np.linalg.det(sigma))) * np.exp(
            -0.5 * (x-mu).dot(np.linalg.inv(sigma)).dot((x-mu).T))


    def expectation(self):

        for i in range(self.SIZE):
            for j in range(self.K):
                self.p_clus[i][j] = self.weights[j] * self.pdf(sigma=self.sigma[j], mu=self.mu[j], x=self.X[[i]])
        self.p_clus = self.p_clus / self.p_clus.sum(axis=0,keepdims=True)

    def maximisation(self):
        for i in range(self.K):
            p = self.p_clus[:, i]
            self.mu[i] = np.sum(p * self.X, axis=0) / np.sum(p)
            self.weights[i] = np.sum(p) / self.SIZE
            self.sigma[i] = (1 / np.sum(p)) * np.sum(p * (self.X - self.mu[i]) * np.transpose(self.X - self.mu[i]))

    def start(self):
        ex = 0
        while ex < self.MAX_ITR:
            self.expectation()
            self.maximisation()
            ex += 1
        print self.mu


if __name__ == '__main__':
    g = GMM()
    g.start()
