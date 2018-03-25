from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Import MNIST data
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.examples.tutorials.mnist import input_data

from HW3 import DATA_DIR


def get_ng_vectors(mode='train'):
    NG_DATA = fetch_20newsgroups(data_home=DATA_DIR, subset=mode, remove=('headers', 'footers', 'quotes'), )
    labels = NG_DATA.target

    #  pca for dimension reduction
    vec = TfidfVectorizer(min_df=2, max_df=0.5, stop_words='english').fit_transform(NG_DATA.data).todense()
    # pca = PCA(n_components=5000)
    # pca.fit(vec, labels)

    # transform to 5000 inputs
    # labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # vec = tf.convert_to_tensor(vec, dtype=tf.float32)
    rows = 15000
    if mode == 'test':
        rows = 5000
    return vec[:rows], labels[:rows]


vec, labels = get_ng_vectors()
# Parameters
learning_rate = 0.1
num_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 1000  # 1st layer number of neurons
n_hidden_2 = 1000  # 2nd layer number of neurons
num_input = vec.shape[1]  # MNIST data input (img shape: 28*28)
num_classes = 20  # MNIST total classes (0-20 classes)
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': vec}, y=labels, batch_size=batch_size, num_epochs=None,
                                              shuffle=True)


# Define the neural network
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    logits = neural_net(features)

    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Train the Model
print(model.train(input_fn, steps=num_steps))

vec_test, labels_test = get_ng_vectors()
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': vec_test}, y=labels_test, batch_size=batch_size,
                                              shuffle=False)
# Use the Estimator 'evaluate' method
print(model.evaluate(input_fn))

# Predict single images
# n_images = 4
# # Get images from test set
# test_images = mnist.test.images[:n_images]
# # Prepare the input data
# input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={'images': test_images}, shuffle=False)
# # Use the model to predict the images class
# preds = list(model.predict(input_fn))

# # Display
# for i in range(n_images):
#     plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
#     plt.show()
#     print("Model prediction:", preds[i])
