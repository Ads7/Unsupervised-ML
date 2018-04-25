import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

DATA_DIR = '/Users/aman/Dropbox/CS6220_Amandeep_Singh/Exam/PB1/mnist_noisy_SAMPLE5000_K20_F31.txt'
K = 20
#  loading data
data = np.loadtxt(DATA_DIR)

# print(data.shape)              checking data shape (5000, 785)

# first column is the label (digit), then the other 784 columns are pixel normalized values.
labels = data[:, 0].astype(int)  # print(np.unique(labels)) [0 1 2 3 4 5 6 7 8 9]
# VECTORS = StandardScaler().fit_transform(data[:, 1:])
VECTORS = data[:, 1:]


def cal_purity(predictions, labels, size):
    """
    :return: : returns the purity
    :rtype: float
    """
    purity = 0.0
    k = np.unique(predictions).shape[0]
    for i in range(k):
        indexes = np.argwhere(predictions == i)
        counts = np.bincount(labels[indexes.transpose()[0]])
        if len(counts):
            purity += counts.max()
    purity = purity / size
    return purity


def harmonic_mean(a, b):
    return 2 * a * b / (a + b)


if __name__ == '__main__':

    # Part 1
    pca = PCA(n_components=90, whiten=True, svd_solver='full')
    vectors_ = pca.fit_transform(VECTORS)
    heirachy = AgglomerativeClustering(n_clusters=20)
    labels_pred = heirachy.fit_predict(vectors_)
    p_1 = cal_purity(labels_pred, labels, labels.shape[0])
    p_2 = cal_purity(labels, labels_pred, labels.shape[0])
    mean = harmonic_mean(p_1, p_2)
    print(p_1, p_2)
    print(mean)
    # Part 2
    clf = ExtraTreesClassifier()
    pca = PCA(n_components=90, whiten=True, svd_solver='full')
    vectors_ = pca.fit_transform(VECTORS)

    X = VECTORS[:500, :]
    Y = labels[:500]
    labels_pred = [None] * 5000
    for i in range(5):
        dtc = KNeighborsClassifier(n_neighbors=3)
        dtc.fit(X, Y)
        labels_prob = dtc.predict_proba(VECTORS)
        for i, val in enumerate(labels_prob):
            if not labels_pred[i] and val.max() > .8:
                labels_pred[i] = np.argmax(val)
                labels_pred = np.array(labels_pred)
        index = np.argwhere(labels_pred != None).transpose()[0].tolist()
        X = VECTORS[index, :]
        Y = labels_pred[index].astype('int')
    index = np.argwhere(labels_pred == None).transpose()[0].tolist()
    if index:
        heirachy = AgglomerativeClustering(n_clusters=20)
        labels_pred_h = heirachy.fit_predict(vectors_)
        labels_pred[index] = labels_pred_h[index]
    labels_pred = np.array(labels_pred).astype('int')
    p_1 = cal_purity(labels_pred, labels, labels.shape[0])
    p_2 = cal_purity(labels, labels_pred, labels.shape[0])
    mean = harmonic_mean(p_1, p_2)
    print(p_1, p_2)
    print(mean)
    # x.append(k)
    # y.append(mean)
    # plt.plot(x, y)
    # plt.savefig("test")
