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
    attr = ['x', 'y']

    def __init__(self, min_pt=3, esp=0.3):
        """

        :param min_pt: int value of min neighbours each point must have to be a core point
        :param esp: float value of radius of scope of each point
        """
        self.min_pt = min_pt
        self.esp = esp

    def fit(self, vectors):
        """

        :type vectors: pandas object having attrs and a column cluster(labels)
        """
        # todo can seperate the vectors as numpy array and labels
        # todo update code as per numpy
        dist_matrix = euclidean_distances(vectors[self.attr].values, vectors[self.attr].values)
        C = 1
        for i, pt in vectors.iterrows():
            if vectors.at[i, 'cluster'] == NOISE or np.isnan(vectors.at[i, 'cluster']):
                index = np.where(dist_matrix[i] < self.esp)[0].tolist()  # correct code get index
                if len(index) < self.min_pt:
                    vectors.at[i, 'cluster'] = NOISE
                else:
                    vectors.at[i, 'cluster'] = C
                    q = index
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
        """

        :param vectors: pandas object having labels and vector values
        :param name: name for the image
        """
        # todo visualisation to be updated as for numpy
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
        plt.savefig("img/" + name)

    def cal_purity(self, K, prediction, labels):
        """
        :return: : returns the purity and the gini coeff
        :rtype: float,float
        """
        # cal_purity(gmm.predict(FMNIST[:, :-1]), 2, FMNIST[:, [-1]], FMNIST[:, [-1]].shape[0], True)
        purity = 0.0
        SIZE = labels.shape[0]
        for i in range(K):
            indexes = np.argwhere(prediction == i)
            counts = np.bincount(labels[indexes.transpose()[0]])
            purity += counts.max()
        purity = purity / SIZE
        return purity


class TOYDBSCAN(DBSCAN):

    def read(self, path):
        data = pd.read_csv(DATA_DIR + path)
        data['cluster'] = np.nan
        return data


# todo complete working of household, fashion and NG20
class HOUSEDATA(DBSCAN):
    # Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;
    # Sub_metering_3
    attr = ['x', 'Global_reactive_power', 'Voltage',
            'Global_intensity']

    def read(self, path):
        # .(global_active_power * 1000 / 60 - sub_metering_1 - sub_metering_2 - sub_metering_3)
        data = pd.read_csv(DATA_DIR + path, sep=';')
        data['x'] = data['Global_active_power'] * 1000 / 60 - (
                data['Sub_metering_1'] + data['Sub_metering_2'] + data['Sub_metering_3'])
        data['cluster'] = np.nan
        return data


class NG20(DBSCAN):
    # Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;
    # Sub_metering_3
    attr = ['x', 'Global_reactive_power', 'Voltage',
            'Global_intensity']

    def read(self, path):
        data = pd.read_csv(DATA_DIR + path)
        data['cluster'] = np.nan
        return data


class FASHIOND(DBSCAN):
    # Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;
    # Sub_metering_3
    attr = ['x', 'Global_reactive_power', 'Voltage',
            'Global_intensity']

    def read(self, path):
        # .(global_active_power * 1000 / 60 - sub_metering_1 - sub_metering_2 - sub_metering_3)
        data = pd.read_csv(DATA_DIR + path, sep=';')
        data['x'] = data['Global_active_power'] * 1000 / 60 - (
                data['Sub_metering_1'] + data['Sub_metering_2'] + data['Sub_metering_3'])
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
    dbs.visualise(vectors, 'moon.png')
    print("DBSCAN circle data")
    dbs = TOYDBSCAN(min_pt=2, esp=0.18)
    vectors = dbs.fit(dbs.read('/circle.csv'))
    dbs.visualise(vectors, 'circle.png')
    print("DBSCAN blob data")
    dbs = TOYDBSCAN(esp=.37)
    vectors = dbs.fit(dbs.read('/blobs.csv'))
    dbs.visualise(vectors, 'blobs.png')
