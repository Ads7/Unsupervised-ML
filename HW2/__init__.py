from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import fetch_20newsgroups
HW_DIR = ''
DATA_DIR = '/Users/aman/workspace/DataMining/'+HW_DIR
MNIST = input_data.read_data_sets(DATA_DIR+"MNIST_data/", one_hot=True)
NEWS_DATA = fetch_20newsgroups(data_home=DATA_DIR,subset='test', remove=('headers', 'footers', 'quotes'),)





