import scipy
from scipy import sparse, random
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from HW2 import NEWS_DATA, MNIST


class Kmeans(object):
    K = None
    centroids = []
    vectors = None
    TOLERANCE = 0.01
    SIZE = None

    def __init__(self, data=None, k=10):
        self.K = k
        self.initialize()
        self.start()

    def start(self):
        self.create_clusters()

    def array_to_image(self, X, path):
        X.reshape([28, 28])
        scipy.misc.imsave(path, X)

    def initialize(self):
        self.vectors = MNIST.train.images
        self.SIZE = self.vectors.shape[0]
        for x in range(10):
            self.centroids.append(self.vectors[random.randint(1, self.SIZE)])
            # vectorizer = TfidfVectorizer()
            # vectors = vectorizer.fit_transform(NEWS_DATA.data)
            # cos_sim = cosine_similarity(X=sparse.csr_matrix(vectors.toarray()), Y=sparse.csr_matrix(vectors.toarray()))
            # random 10 numbers from 0 to number of rows

    def create_clusters(self):
        ex =0
        #  todo add stopping crite
        while ex <85:
            new_clusters = []
            distances = euclidean_distances(self.vectors, self.centroids).argmin(axis=1)
            for i in range(self.K):
                indexes = np.argwhere(distances == i)
                data = self.vectors[indexes.transpose()[0]]
                new_clusters.append(np.sum(data, axis=0) / data.shape[0])
            ex +=1
            self.centroids = new_clusters

        for i in range(self.K):
            plt.figure()
            plt.imshow(new_clusters[i].reshape([28, 28]),cmap='gray')
        plt.show()



# assign to cluster
#       take mean reinitialise centroid
#       repeat
#       calculate purity
#



if __name__ == '__main__':
    Kmeans()
