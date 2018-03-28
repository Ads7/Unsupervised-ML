from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.contrib.metrics import accuracy

from HW2.kmeans import Kmeans
from HW3 import DATA_DIR
from HW3.classification import Classify


def get_ng_vectors(mode='train'):
    NG_DATA = fetch_20newsgroups(data_home=DATA_DIR, subset=mode, remove=('headers', 'footers', 'quotes'), )
    labels = NG_DATA.target
    vec = TfidfVectorizer(min_df=2, max_df=0.5, stop_words='english').fit_transform(NG_DATA.data).todense()
    return vec, labels


vec, labels = get_ng_vectors()
labels_hot = np.eye(20)[labels]
X_train, X_test, y_train, y_test = train_test_split(vec, labels, test_size=0.4, random_state=0)
y_train_hot = np.eye(20)[y_train]
y_test_hot = np.eye(20)[y_test]
# Training Parameters
learning_rate = 0.01
num_steps = 1000
batch_size = 256

display_step = 100

# Network Parameters
num_hidden_1 = 1000  # 1st layer num features
num_hidden_2 = 500  # 2nd layer num features (the latent dim)
num_input = X_train.shape[1]  # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])


print(X_test.shape)
# Kmeans(X_test, y_test,20)

for layer in [ 500]:
    # Initialize the variables (i.e. assign their default value)
    num_hidden_1 = 500
    num_hidden_2 = 500
    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
        # 'encoder_h2':  tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
        # 'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
        # 'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
        'decoder_b1': tf.Variable(tf.random_normal([num_input])),
        # 'decoder_b2': tf.Variable(tf.random_normal([num_input])),
    }


    # Building the encoder
    def encoder(x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        # Encoder Hidden layer with sigmoid activation #2
        # layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
        #                                biases['encoder_b2']))
        return layer_1


    # Building the decoder
    def decoder(x):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        # layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
        #                                biases['decoder_b2']))
        return layer_1


    # Construct model
    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

    # Prediction
    y_pred = decoder_op
    # Targets (Labels) are the input data.
    y_true = X

    # Define loss and optimizer, minimize the squared error
    loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    # Start Training
    # Start a new TF session
    sess = tf.Session()

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        start = i % X_train.shape[0]
        batch_x, _ = X_train[start: start + batch_size], y_train_hot[start: start + batch_size]
        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    c = Classify(sess.run(encoder_op, feed_dict={X: X_train}), y_train, sess.run(encoder_op, feed_dict={X: X_test}),
                 y_test)
    c.lr(sess.run(encoder_op, feed_dict={X: X_train}), y_train, sess.run(encoder_op, feed_dict={X: X_test}), y_test)

# Step 1: Minibatch Loss: 0.469407
# Step 100: Minibatch Loss: 0.068125
# Step 200: Minibatch Loss: 0.052881
# Step 300: Minibatch Loss: 0.044515
# Step 400: Minibatch Loss: 0.039488
# Step 500: Minibatch Loss: 0.036451
# Step 600: Minibatch Loss: 0.034068
# Step 700: Minibatch Loss: 0.032457
# Step 800: Minibatch Loss: 0.031130
# Step 900: Minibatch Loss: 0.029700
# accuracy:  0.418692001768

