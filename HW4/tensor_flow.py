# Import MNIST data
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
from tensorflow.examples.tutorials.mnist import input_data

from HW2 import DATA_DIR
from HW3.classification import get_ng_vectors

mnist = input_data.read_data_sets(DATA_DIR + "MNIST_data/", one_hot=True)
vec, labels = get_ng_vectors()
import tensorflow as tf

data_sets = [dict(key='20NG',
                  classes=20,
                  features=5000,
                  data=mnist),
             dict(key='MNIST',
                  classes=10,
                  features=784,
                  data=mnist)]
# Set parameters
learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2

for data_set in data_sets:
    print data_set
    if data_set['key'] == '20NG':
        X_train, X_test, y_train, y_test = train_test_split(vec[:5000], labels[:5000], test_size=0.4, random_state=0)
    else:
        X_train, X_test, y_train, y_test = mnist.train.images, mnist.test.images, mnist.train.labels, mnist.test.labels
        # TF graph input
    x = tf.placeholder("float", [None, data_set['features']])  #  data image of shape 28*28=784
    y = tf.placeholder("float", [None, data_set['classes']])  # 0-9 digits recognition => 10 classes

    # Create a model

    # Set model weights
    W = tf.Variable(tf.zeros([data_set['features'], data_set['classes']]))
    b = tf.Variable(tf.zeros([data_set['classes']]))

    with tf.name_scope("Wx_b") as scope:
        # Construct a linear model
        model = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

    # Add summary ops to collect data
    w_h = tf.summary.histogram("weights", W)
    b_h = tf.summary.histogram("biases", b)

    # More name scopes will clean up graph representation
    with tf.name_scope("cost_function") as scope:
        # Minimize error using cross entropy
        # Cross entropy
        cost_function = -tf.reduce_sum(y * tf.log(model))
        # Create a summary to monitor the cost function
        tf.summary.scalar("cost_function", cost_function)

    with tf.name_scope("train") as scope:
        # Gradient descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Merge all summaries into a single operator
    merged_summary_op = tf.summary.merge_all()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Change this to a location on your computer
        summary_writer = tf.summary.FileWriter('/Users/aman/Dropbox/CS6220_Amandeep_Singh/HW4', graph=sess.graph)

        # Training cycle
        for iteration in range(training_iteration):
            avg_cost = 0.
            total_batch = int(X_train.shape[0] / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                if data_set['key'] == '20NG':
                    batch_xs, batch_ys = X_train, y_train[i * batch_size:i * batch_size + i * batch_size]
                else:
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # Fit training using batch data
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
                # Compute the average loss
                avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
                # Write logs for each iteration
                summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
                summary_writer.add_summary(summary_str, iteration * total_batch + i)
            # Display logs per iteration step
            if iteration % display_step == 0:
                print "Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost)

        print "Tuning completed!"

        # Test the model
        predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
        print "Accuracy:", accuracy.eval({x: X_test, y: y_test})
