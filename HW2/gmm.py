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
        self.TOLERANCE = 1.e-5
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
        vectors = np.genfromtxt(DATA_DIR + '/' + path, delimiter=' ')
        print "Data loaded"
        return vectors

    # def pdf(self, sigma, mu, x):
    #     stats.multivariate_normal.pdf(x=x,sigma)
    #     return (1.0 / np.sqrt(((2 * np.pi) ** self.dimension / 2) * np.linalg.det(sigma))) * np.exp(
    #         -0.5 * (x-mu).dot(np.linalg.inv(sigma)).dot((x-mu).T))

    def expectation(self):

        for i in range(self.SIZE):
            for j in range(self.K):
                self.p_clus[i][j] = self.weights[j] * stats.multivariate_normal.pdf(x=self.X[[i]], cov=
                self.sigma[j], mean=self.mu[j])
        for i in range(self.SIZE):
            self.p_clus[i] = self.p_clus[i] / self.p_clus[i].sum()

    def maximisation(self):
        for i in range(self.K):
            p = self.p_clus[:, [i]]
            self.mu[i] = np.sum(p * self.X, axis=0) / np.sum(p)
            self.weights[i] = np.sum(p) / self.SIZE

            tmp = self.X - self.mu[i]
            self.sigma[i] = np.dot(np.transpose(tmp), np.multiply(tmp, p)) / np.sum(p)

    def start(self):
        ex = 0
        while ex < self.MAX_ITR:
            old = self.p_clus.copy()
            self.expectation()
            self.maximisation()
            if np.allclose(self.p_clus, old,atol=self.TOLERANCE):
                break
            ex += 1
        pos = np.bincount(np.argmax(self.p_clus,axis=1))
        print 'iterations taken: ',ex
        for i in range(self.K):
            print self.mu[i]
            print self.sigma[i]
            print(pos[i])


if __name__ == '__main__':
    g = GMM()
    g.start()
