# class DBSCAN()
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances

from HW2 import DATA_DIR
from HW2.gmm import cal_purity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABEL = 0
NOISE = 99999


class DBSCAN(object):

    def __init__(self, min_pt= 3, esp=0.3):
        self.min_pt = min_pt
        self.esp = esp

    def fit(self, vectors):
        dist_matrix = euclidean_distances(vectors[['x', 'y']].values, vectors[['x', 'y']].values)
        C = 1
        for i, pt in vectors.iterrows():
            if vectors.at[i, 'cluster'] == NOISE or np.isnan(vectors.at[i, 'cluster']):
                index = np.where(dist_matrix[i] < self.esp)[0].tolist()  # correct code get index
                if len(index) < self.min_pt:
                    vectors.at[i, 'cluster'] = NOISE
                else:
                    vectors.at[i, 'cluster'] = C
                    q = index  # remove the current point here
                    while q:
                        j = q.pop()
                        n_pt = vectors.loc[j]
                        if n_pt['cluster'] == NOISE or np.isnan(n_pt['cluster']):
                            vectors.at[j, 'cluster'] = C
                            q.extend(np.where(dist_matrix[j] < self.esp)[0].tolist())
                C += 1
        return vectors

    def read(self, path):
        return pd.read_csv(DATA_DIR + path)

    def visualise(self, vectors, name='img.png'):
        plt.figure()
        unique_labels = vectors.cluster.unique().tolist()
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == NOISE:
                # Black used for noise.
                col = [0, 0, 0, 1]
            plt.plot(vectors[vectors.cluster == k].x.values.tolist(),
                     vectors[vectors.cluster == k].y.values.tolist(),
                     'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)
        plt.savefig(name)


class TOYDBSCAN(DBSCAN):

    def read(self, path):
        data = pd.read_csv(DATA_DIR + path)
        data['cluster'] = np.nan
        return data


def hierarchical_clustering():
    from tensorflow.examples.tutorials.mnist import input_data
    MNIST = input_data.read_data_sets(DATA_DIR + "MNIST_data/")
    print("Starting hierarchical_clustering")
    hc = AgglomerativeClustering(n_clusters=10)
    hc.fit(MNIST.test.images)
    print cal_purity(hc.labels_, 10, MNIST.test.labels, MNIST.test.images.shape[0])


if __name__ == '__main__':
    print(" starting hierarchical_clustering")
    # hierarchical_clustering()
    print("DBSCAN toy data")
    dbs = DBSCAN(min_pt=3, esp=7.5)
    vectors = dbs.fit(dbs.read('/dbscan.csv'))
    dbs.visualise(vectors)
    print("DBSCAN moon data")
    dbs = TOYDBSCAN(min_pt=2)
    vectors = dbs.fit(dbs.read('/moons.csv'))
    dbs.visualise(vectors,'moon.png')
    print("DBSCAN circle data")
    dbs = TOYDBSCAN(min_pt=2,esp=0.18)
    vectors = dbs.fit(dbs.read('/circle.csv'))
    dbs.visualise(vectors, 'circle.png')
    print("DBSCAN blob data")
    dbs = TOYDBSCAN(esp=.37)
    vectors = dbs.fit(dbs.read('/blobs.csv'))
    dbs.visualise(vectors, 'blobs.png')
