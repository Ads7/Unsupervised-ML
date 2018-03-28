from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Import MNIST data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.examples.tutorials.mnist import input_data

from HW2.kmeans import Kmeans
from HW3 import DATA_DIR
from HW3.classification import Classify

data = pd.read_csv(DATA_DIR + 'spambase/spambase.data', header=None)
data.rename(columns={57: 'is_spam'}, inplace=True)
target = data.pop('is_spam')
vectors = StandardScaler().fit_transform(data).astype(int)

X_train, X_test, y_train, y_test = train_test_split(vectors, target, test_size=0.4, random_state=0)
y_train_hot = np.eye(2)[y_train]
y_test_hot = np.eye(2)[y_test]

# Training Parameters
learning_rate = 0.01
num_steps = 15000
batch_size = 256

display_step = 1000
examples_to_show = 10

# Network Parameters
num_hidden_1 = 25  # 1st layer num features
num_hidden_2 = 10  # 2nd layer num features (the latent dim)
num_input = X_train.shape[1]  # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


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

# Initialize the variables (i.e. assign their default value)
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
    start = i%X_train.shape[0]
    batch_x, _ = X_train[start: start + batch_size], y_train_hot[start: start + batch_size]
    # Run optimization op (backprop) and cost op (to get loss value)
    _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
    # Display logs per step
    if i % display_step == 0 or i == 1:
        print('Step %i: Minibatch Loss: %f' % (i, l))


# g = sess.run(encoder_op, feed_dict={X: X_test})
c = Classify(sess.run(encoder_op, feed_dict={X: X_train}), y_train,sess.run(encoder_op, feed_dict={X: X_test}), y_test)
c.lr(sess.run(encoder_op, feed_dict={X: X_train}), y_train,sess.run(encoder_op, feed_dict={X: X_test}), y_test)
# Kmeans(g, y_train,2)

# Step 1: Minibatch Loss: 1.037036
# Step 1000: Minibatch Loss: 0.442296
# Step 2000: Minibatch Loss: 0.487774
# Step 3000: Minibatch Loss: 0.674182
# Step 4000: Minibatch Loss: 0.556992
# Step 5000: Minibatch Loss: 0.457380
# Step 6000: Minibatch Loss: 0.695595
# Step 7000: Minibatch Loss: 0.559287
# Step 8000: Minibatch Loss: 0.605949
# Step 9000: Minibatch Loss: 0.872496
# Step 10000: Minibatch Loss: 0.495103
# Step 11000: Minibatch Loss: 0.393013
# Step 12000: Minibatch Loss: 0.780972
# Step 13000: Minibatch Loss: 0.582411
# Step 14000: Minibatch Loss: 0.690287
# accuracy:  0.813145029875
