from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

from HW2 import DATA_DIR
from HW2.gmm import cal_purity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NOISE = 99999
DICT_DATA = {
    "toy": dict(
        path="/dbscan.csv",
        esp=7.5,
    ),
    "moons": dict(
        path='/moons.csv',
        min_pt=2
    ),
    "circles": dict(
        path="/circle.csv",
        esp=0.18,
        min_pt=2
    ),
    "blobs": dict(
        path="/blobs.csv",
        esp=0.37,
    )
}


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

        dist_matrix = euclidean_distances(vectors, vectors)
        C = -1
        labels = np.empty((vectors.shape[0], 1))
        labels[:] = np.nan
        for i in range(vectors.shape[0]):
            if labels[i] == NOISE or np.isnan(labels[i]):
                index = np.where(dist_matrix[i] < self.esp)[0].tolist()  # correct code get index
                if len(index) < self.min_pt:
                    labels[i] = NOISE
                else:
                    C += 1
                    labels[i] = C
                    q = index
                    while q:
                        j = q.pop()
                        n_pt = labels[j]
                        if n_pt == NOISE or np.isnan(n_pt):
                            labels[j] = C
                            q.extend(np.where(dist_matrix[j] < self.esp)[0].tolist())
        return labels

    def read(self, path):
        return pd.read_csv(DATA_DIR + path)[self.attr].values

    def visualise(self, labels, vectors, name='img.png'):
        """
        Only for two attribute datas
        :param vectors: pandas object having labels and vector values
        :param name: name for the image
        """
        plt.figure()
        unique_labels = np.unique(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == NOISE:
                # Black used for noise.
                col = [0, 0, 0, 1]
            plt.plot(vectors[np.where(labels == k), 0].tolist(),
                     vectors[np.where(labels == k), 1].tolist(),
                     'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)
        plt.savefig("img/" + name)

    def cal_purity(self, prediction, labels):
        """
        :return: : returns the purity and the gini coeff
        :rtype: float,float
        """
        # cal_purity(gmm.predict(FMNIST[:, :-1]), 2, FMNIST[:, [-1]], FMNIST[:, [-1]].shape[0], True)
        purity = 0.0
        K = np.unique(prediction).shape[0] - 1
        print(K)
        SIZE = labels.shape[0]
        for i in range(K):
            indexes = np.argwhere(prediction == i)
            counts = np.bincount(labels[indexes.transpose()[0]])
            purity += counts.max()
        purity = purity / SIZE
        print purity
        return purity


# todo complete working of household
class HOUSEDATA(DBSCAN):
    # Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;
    # Sub_metering_3
    attr = ['Global_reactive_power', 'Voltage',
            'Global_intensity']

    def read(self, path):
        # .(global_active_power * 1000 / 60 - sub_metering_1 - sub_metering_2 - sub_metering_3)
        data = pd.read_csv(DATA_DIR + path, sep=';')
        for att in self.attr + ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2']:
            data = data[data[att] != '?']
            data[att] = pd.to_numeric(data[att])
        self.attr.append('x')
        data['x'] = data['Global_active_power'] * 1000 / 60 - (
                data['Sub_metering_1'] + data['Sub_metering_2'] + data['Sub_metering_3'])
        return data[self.attr].values[:10000, :]


class NG20(DBSCAN):

    def read(self, path=None):
        NEWS_DATA = fetch_20newsgroups(data_home=DATA_DIR, subset='test', remove=('headers', 'footers', 'quotes'), )
        return TfidfVectorizer(stop_words='english', min_df=3, max_df=0.6).fit_transform(
            NEWS_DATA.data).todense(), NEWS_DATA.target


class FASHIOND(DBSCAN):
    def read(self, path=None):
        FMNIST = np.genfromtxt(DATA_DIR + '/fashionmnist/fashion-mnist_test.csv', delimiter=',')
        print "Data loaded"
        labels = FMNIST[1:-1, 0].astype(int)
        vectors = FMNIST[1:-1, 1:]
        return vectors, labels


def hierarchical_clustering():
    from tensorflow.examples.tutorials.mnist import input_data
    MNIST = input_data.read_data_sets(DATA_DIR + "MNIST_data/")
    print("Starting hierarchical_clustering")
    hc = AgglomerativeClustering(n_clusters=10)
    hc.fit(MNIST.test.images)
    print cal_purity(hc.labels_, 10, MNIST.test.labels, MNIST.test.images.shape[0])


if __name__ == '__main__':
    for key in DICT_DATA:
        print("DBSCAN for data: " + key)
        dict_val = DICT_DATA[key]
        dbs = DBSCAN(min_pt=dict_val.get('min_pt', 3), esp=dict_val.get('esp', 0.3))
        vectors = dbs.read(dict_val.get('path'))
        labels = dbs.fit(vectors)
        dbs.visualise(labels, vectors, key + '.png')

    print("DBSCAN NG20")
    dbs = NG20(min_pt=3, esp=.3)
    vectors, labels = dbs.read()
    pred_labels = dbs.fit(vectors)
    print silhouette_score(vectors, pred_labels.transpose()[0])
    dbs.cal_purity(pred_labels, labels)
    print("DBSCAN FASHIOND")
    dbs = FASHIOND(min_pt=2, esp=.3)
    vectors, labels = dbs.read()
    pred_labels = dbs.fit(vectors)
    print silhouette_score(vectors, pred_labels.transpose()[0])
    dbs.cal_purity(pred_labels, labels)
    print("DBSCAN HouseHold")
    dbs = HOUSEDATA(min_pt=2, esp=.3)
    vectors = dbs.read('/household_power_consumption.txt')
    pred_labels = dbs.fit(vectors)
    print silhouette_score(vectors, pred_labels.transpose()[0])
    print(" starting hierarchical_clustering")
    hierarchical_clustering()
