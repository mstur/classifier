from __future__ import print_function
from datetime import datetime
import get_data
import argparse
import sys
import tensorflow as tf

PIXELS = 784
NUM_CLASSES = 10
FLAGS = None
VALIDATION_SIZE = 5000
BATCH_SIZE = 100
LEARNING_RATE = 0.01
TRAINING_EPOCHS = 25


# creates bias, initializes with small positive value to avoid "dead neurons"
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # sets up tensorboard logging
    now=datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    data = get_data.read_data_sets(FLAGS.data_dir, PIXELS, NUM_CLASSES, validation_size=VALIDATION_SIZE)

    # ______________________________________CONSTRUCTION PHASE______________________________________
    # graph input
    x = tf.placeholder(tf.float32, [None, PIXELS])
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # model weights
    W = tf.Variable(tf.zeros([PIXELS, NUM_CLASSES]))     # weights
    b = tf.Variable(tf.zeros([NUM_CLASSES]))    # bias

    # example of name scopes (can group related nodes)
    with tf.name_scope('Model'):
        predicted = tf.nn.softmax(tf.matmul(x, W)+b)
    with tf.name_scope('Loss'):
        cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(predicted), reduction_indices=1))
    with tf.name_scope('SGD'):
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)
    with tf.name_scope('Accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predicted, 1), tf.argmax(y, 1)), tf.float32))

    tf.summary.scalar("loss", cost)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    '''
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    ce_summary = tf.summary.scalar('CE', cross_entropy)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    '''
    saver = tf.train.Saver()
    # ________________________________________EXECUTION PHASE_______________________________________
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver.restore(sess, "/tmp/easy_final.ckpt")
    summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
    # Train
    avg_cost = 0
    for epoch in range(TRAINING_EPOCHS):
        total_batch = int(data.train.num_examples / BATCH_SIZE)
        if epoch % 5 == 0:
            save_path = saver.save(sess, "/tmp/easy.ckpt")
            print("Progress Saved!")
        for batchn in range(total_batch):
            batch = data.train.next_batch(BATCH_SIZE)
            _ = sess.run(optimizer, feed_dict={x: batch[0], y: batch[1]})
            c = sess.run(cost, feed_dict={x: batch[0], y: batch[1]})
            summary = sess.run(merged_summary_op, feed_dict={x: batch[0], y: batch[1]})
            summary_writer.add_summary(summary, epoch * total_batch + batchn)
            avg_cost += c / total_batch
        print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost))

    # Test trained model
    print("Optimization Finished!")
    print("Accuracy: ", sess.run(accuracy, feed_dict={x: data.test.images, y: data.test.labels}))

    saver.save(sess, "/tmp/easy_final.ckpt")
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/', help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
