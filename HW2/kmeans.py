import scipy
from scipy import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.examples.tutorials.mnist import input_data

from HW2 import DATA_DIR

TEST = 'test'
TRAIN = 'train'


class Kmeans(object):
    K = None
    MAX_ITERATION = 100
    centroids = []
    labels = None
    vectors = None
    TOLERANCE = 1.e-5
    SIZE = None
    is_distance_based = True
    mode = TEST

    def __init__(self, k=10, mode=TEST):
        """
        :param k(int): number of clusters to consider default 10
        """
        self.K = k
        self.mode = mode
        self.initialize()
        self.start()

    def start(self):
        self.create_clusters()

    def init_vectors(self):  # can be replaced with abstract decorator in python3
        """
            :return numpy array: vectors of the dataset
        """
        raise NotImplementedError

    def initialize(self):
        """
            to initialise vectors, its size and randomly allocated centroids
        """
        self.init_vectors()
        self.SIZE = self.vectors.shape[0]
        # todo can use max distance to allocation farthest apart points
        self.centroids = self.vectors[[random.randint(1, self.SIZE) for x in range(self.K)], :]

    def create_clusters(self):
        """
        Create and update clusters till max iterations or the if change rate drops
        :rtype: None

        """
        ex = 0
        print 'Iter -   Purity               Gini Index'
        while ex < self.MAX_ITERATION:
            new_clusters = np.zeros(self.centroids.shape)
            distances = euclidean_distances(self.vectors, self.centroids).argmin(axis=1)
            for i in range(self.K):
                indexes = np.argwhere(distances == i)
                data = self.vectors[indexes.transpose()[0]]
                if data.shape[0] > 1:
                    new_clusters[i] = (np.sum(data, axis=0) / data.shape[0])
                else:
                    new_clusters[i] = np.sum(data, axis=0)
            print ex, '----', self.cal_purity()
            ex += 1
            if np.allclose(self.centroids, new_clusters, atol=self.TOLERANCE):
                break
            self.centroids = new_clusters

    def cal_purity(self):
        """
        :return: : returns the purity and the gini coeff
        :rtype: float,float
        """
        distances = euclidean_distances(self.vectors, self.centroids).argmin(axis=1)
        purity = 0.0
        gini_coef = 0.0
        for i in range(self.K):
            indexes = np.argwhere(distances == i)
            weight = 1.0 * indexes.shape[0] / self.SIZE
            counts = np.bincount(self.labels[indexes.transpose()[0]])
            gini_index = 1 - np.square(1.0 * counts / indexes.shape[0]).sum()
            gini_coef += weight * gini_index
            purity += counts.max()
        purity = purity / self.SIZE
        return purity, gini_coef


class MNIST_Kmeans(Kmeans):
    @staticmethod
    def array_to_image(X, path):
        """

       :param X: a 784 column numpy array
       :param path: file location to save the file
        """
        X.reshape([28, 28])
        scipy.misc.imsave(path, X)

    def plot_centroids(self):
        for i in range(self.K):
            plt.figure()
            plt.imshow(self.centroids[i].reshape([28, 28]), cmap='gray')
        plt.show()

    def init_vectors(self):
        MNIST = input_data.read_data_sets(DATA_DIR + "MNIST_data/")
        self.labels = MNIST.train.labels if self.mode == TRAIN else MNIST.test.labels
        self.vectors = MNIST.train.images if self.mode == TRAIN else MNIST.test.images


class NG_Kmeans(Kmeans):
    def init_vectors(self):
        NEWS_DATA = fetch_20newsgroups(data_home=DATA_DIR, subset=self.mode, remove=('headers', 'footers', 'quotes'), )
        self.labels = NEWS_DATA.target
        min_df = 3
        max_df = 0.6
        if self.mode == TRAIN:
            min_df = 4
            max_df = 0.8
        self.vectors = TfidfVectorizer(stop_words='english', min_df=min_df, max_df=max_df).fit_transform(
            NEWS_DATA.data).todense()


class FMNIST_Kmeans(Kmeans):
    @staticmethod
    def array_to_image(X, path):
        """

        :param X: a 784 column numpy array
        :param path: file location to save the file
        """
        X.reshape([28, 28])
        scipy.misc.imsave(path, X)

    def init_vectors(self):
        print "Loading data"
        FMNIST = np.genfromtxt(DATA_DIR + '/fashionmnist/fashion-mnist_' + self.mode + '.csv', delimiter=',')
        print "Data loaded"
        self.labels = FMNIST[1:-1, 0].astype(int)
        self.vectors = FMNIST[1:-1, 1:]


if __name__ == '__main__':
    print "Starting with MNSIT data"
    print 'mode: test'
    MNIST_Kmeans()
    print "------------------------"
    print 'mode: train'
    MNIST_Kmeans(mode=TRAIN)
    print "------------------------"
    print "Starting with news group data"
    print 'mode: test'
    NG_Kmeans(20)
    print "------------------------"
    print 'mode: train'
    NG_Kmeans(k=20, mode=TRAIN)
    print "------------------------"
    print "Starting with fashion data"
    print 'mode: test'
    FMNIST_Kmeans()
    print "------------------------"
    print 'mode: train'
    FMNIST_Kmeans(mode=TRAIN)
    print "------------------------"
