from random import randint

import pandas as pd
from sklearn import linear_model, tree, feature_selection
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data

from HW3 import DATA_DIR


class Classify(object):
    top = 30

    def __init__(self, X, Y, x_test, y_test, top=30):
        self.X = X
        self.Y = Y
        self.x_test = x_test
        self.y_test = y_test
        self.top = top

    def lr(self, X, Y, x_test, y_test, lr=linear_model.LogisticRegression()):
        lr.fit(X, Y)
        labels = lr.predict(x_test)
        print "accuracy: ", accuracy_score(y_test, labels)

    def lasso(self):
        lr = linear_model.Lasso()
        self.lr(self.X, self.Y, self.x_test, self.y_test, lr)
        index = []
        for row in range(lr.coef_.shape[0]):
            index.extend(np.argpartition(lr.coef_[0], -self.top)[-self.top:])
        index = list(set(index))
        self.lr(self.X[:, index], self.Y, self.x_test[:, index], self.y_test, lr)

    def logistic_regression(self, penalty='l2'):
        lr = linear_model.LogisticRegression(penalty=penalty)
        self.lr(self.X, self.Y, self.x_test, self.y_test, lr)
        index = []
        for row in range(lr.coef_.shape[0]):
            index.extend(np.argpartition(lr.coef_[0], -self.top)[-self.top:])
        index = list(set(index))
        self.lr(self.X[:, index], self.Y, self.x_test[:, index], self.y_test, lr)

    def decision_tree(self, param=None):
        if param:
            clf = tree.DecisionTreeClassifier(max_features=30)
        else:
            clf = tree.DecisionTreeClassifier()
        clf.fit(self.X, self.Y)
        labels = clf.predict(self.x_test)
        print "accuracy: ", accuracy_score(self.y_test, labels)

    def decision_tree_param_tuner(self):
        self.decision_tree()
        self.decision_tree(dict(max_features=30))

    def pca(self, D=(5, 20)):
        for d in D:
            pca = PCA(n_components=d)
            X_new = pca.fit(self.X, self.Y)
            self.lr(X_new, self.Y, pca.transform(self.x_test), self.y_test)

    def start(self):
        print("logistic regression")
        self.logistic_regression()
        print("decision tree")
        self.decision_tree_param_tuner()

    def custom_pca(self):
        cov_mat = np.cov(self.X.T)
        # cov mat of X vs X.T have diff time
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        for d in [5, 20]:
            # todo change to generic
            w = np.hstack([x[1].reshape(784, 1) for x in eig_pairs[:d]])
            self.lr(self.X.dot(w), self.Y, self.x_test.dot(w), self.y_test)

    def chi_sq(self):
        func = feature_selection.SelectKBest(score_func=feature_selection.chi2, k=200)
        func = func.fit(self.X, self.Y)
        self.lr(func.transform(self.X), self.Y, func.transform(self.x_test), self.Y)
        func = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_classif, k=200)
        func = func.fit(self.X, self.Y)
        self.lr(func.transform(self.X), self.Y, func.transform(self.x_test), self.Y)


def get_ng_vectors(mode='train'):
    NG_DATA = fetch_20newsgroups(data_home=DATA_DIR, subset=mode, remove=('headers', 'footers', 'quotes'), )
    labels = NG_DATA.target
    vec = TfidfVectorizer(stop_words='english').fit_transform(NG_DATA.data).todense()
    return vec, labels


def haar_feature_extractor():
    recs = randint(1, 100)


def spam_base():
    data = pd.read_csv(DATA_DIR + 'spambase/spambase.data', header=None)
    data.rename(columns={57: 'is_spam'}, inplace=True)
    target = data.pop('is_spam')
    # todo double check the working
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state=0)
    return X_train.values, y_train.values, X_test.values, y_test.values


if __name__ == '__main__':
    # print("PROBLEM 1: Supervised Classification")
    # print("MNIST")
    # MNIST = input_data.read_data_sets(DATA_DIR + "MNIST_data/")
    # c = Classify(MNIST.train.images, MNIST.train.labels, MNIST.test.images, MNIST.test.labels)
    # c.start()
    # print("20 NG")
    # x_train, y_train = get_ng_vectors()
    # x_test, y_test = get_ng_vectors()
    # c = Classify(x_train, y_train, x_test, y_test, 1000)
    # c.start()
    # print("Spambase")
    # c = Classify(*spam_base())
    # c.start()
    # print("PROBLEM 2 : PCA library on MNIST")
    # print("A")
    # print("MNIST")
    # c = Classify(MNIST.train.images, MNIST.train.labels, MNIST.test.images, MNIST.test.labels)
    # c.pca()
    # print("Spambase")
    # c = Classify(*spam_base())
    # c.pca(D=[10,20,30,40,50,60])
    # print("PROBLEM 3 : Implement PCA on MNIST")
    # print("MNIST")
    # MNIST = input_data.read_data_sets(DATA_DIR + "MNIST_data/")
    # c = Classify(MNIST.train.images, MNIST.train.labels, MNIST.test.images, MNIST.test.labels)
    # c.custom_pca()
    # print("PROBLEM 4 : Pairwise Feature selection for text")
    x_train, y_train = get_ng_vectors()
    x_test, y_test = get_ng_vectors()
    c = Classify(x_train, y_train, x_test, y_test, 200)
    # c.chi_sq()
    print("l1")
    c.lasso()
