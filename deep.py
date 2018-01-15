from __future__ import print_function
from datetime import datetime
import get_data
import argparse
import sys
import tensorflow as tf

# Data Parameters
FLAGS = None
PIXELS = 784
NUM_CLASSES = 10
VALIDATION_SIZE = 5000

# Training Parameters
BATCH_SIZE = 50
LEARNING_RATE = 0.01
TRAINING_EPOCHS = 40

# DNN Parameters
N_LAY1 = 300
N_LAY2 = 300
N_LAY3 = 300
N_LAY4 = 300
N_LAY5 = 300
N_LAY6 = 300

# Distributing Parameters
SPLITNUM = PIXELS/2

def main(_):
    # sets up tensorboard logging
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "logs/deep"
    logdir = "{}/run-{}/".format(root_logdir, now)

    data = get_data.read_data_sets(FLAGS.data_dir, PIXELS, NUM_CLASSES, validation_size=VALIDATION_SIZE, onehot=False)

    # sets up distributing
    # task_number = int(sys.argv[1])
    # create_worker(task_number)
    cluster = tf.train.ClusterSpec({"local": ["172.21.109.1","172.21.109.2"]})


    # ______________________________________CONSTRUCTION PHASE______________________________________
    # graph input
    x = tf.placeholder(tf.float32, shape=(None, PIXELS), name="x")
    y = tf.placeholder(tf.int64, shape=None, name="y")

    # creates dnn layers
    ''' with tf.name_scope("dnn"):
        hidlay1 = tf.layers.dense(x, N_LAY1, name="hidlay1", activation=tf.nn.relu)
        hidlay2 = tf.layers.dense(hidlay1, N_LAY2, name="hidlay2", activation=tf.nn.relu)
        hidlay3 = tf.layers.dense(hidlay2, N_LAY3, name="hidlay3", activation=tf.nn.relu)
        outputs = tf.layers.dense(hidlay3, NUM_CLASSES, name="outputs") '''
    with tf.name_scope("dnn1"):
        with tf.device("/job:local/task:0"):
            first_batch = tf.slice(x, [0,0], [SPLITNUM, 0])
            hidlay1 = tf.layers.dense(x, N_LAY1, name="hidlay1", activation=tf.nn.relu)
            hidlay2 = tf.layers.dense(hidlay1, N_LAY2, name="hidlay2", activation=tf.nn.relu)
            hidlay3 = tf.layers.dense(hidlay2, N_LAY3, name="hidlay3", activation=tf.nn.relu)
            outputs1 = tf.layers.dense(hidlay3, NUM_CLASSES, name="outputs1")
    with tf.name_scope("dnn2"):
        with tf.device("/job:local/task:1"):
            second_batch = tf.slice(x, [SPLITNUM, 0], [-1,-1])
            hidlay4 = tf.layers.dense(x, N_LAY4, name="hidlay4", activation=tf.nn.relu)
            hidlay5 = tf.layers.dense(hidlay4, N_LAY5, name="hidlay5", activation=tf.nn.relu)
            hidlay6 = tf.layers.dense(hidlay5, N_LAY6, name="hidlay6", activation=tf.nn.relu)
            outputs2 = tf.layers.dense(hidlay6, NUM_CLASSES, name="outputs2")
            outputs=(outputs1+outputs2)/2

    # name scopes group related nodes
    with tf.name_scope('Loss'):
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=outputs)
        loss = tf.reduce_mean(xent, name="loss")
    with tf.name_scope('SGD'):
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        training_op = optimizer.minimize(loss)
    with tf.name_scope('Accuracy'):
        correct = tf.nn.in_top_k(outputs, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()

    # ________________________________________EXECUTION PHASE_______________________________________
    # sess = tf.InteractiveSession()
    # tf.global_variables_initializer().run()
    with tf.Session("grpc://172.21.109.1") as sess:
        # restores model from disk
        # saver.restore(sess, "/tmp/deep_final.ckpt")
        summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
    
        # Train
        for epoch in range(TRAINING_EPOCHS):
            total_batch = int(data.train.num_examples / BATCH_SIZE)
            for batchn in range(total_batch):
                batch = data.train.next_batch(BATCH_SIZE)
                _ = sess.run(training_op, feed_dict={x: batch[0], y: batch[1]})
                summary = sess.run(merged_summary_op, feed_dict={x: batch[0], y: batch[1]})
                summary_writer.add_summary(summary, epoch * total_batch + batchn)
            acc_train = accuracy.eval(feed_dict={x: batch[0], y: batch[1]})
            acc_val = accuracy.eval(feed_dict={x: data.validation.images, y: data.validation.labels})
            print((epoch + 1), "Training accuracy: ", acc_train, "Validation accuracy: ", acc_val)
            if (epoch + 1) % 5 == 0:
                save_path = saver.save(sess, "/tmp/deep.ckpt")
                print("Progress Saved!")
    
        # Test trained model
        print("Training Finished!")
        print("Accuracy: ", sess.run(accuracy, feed_dict={x: data.test.images, y: data.test.labels}))
    
        saver.save(sess, "/tmp/deep_final.ckpt")
        sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/', help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
'''
    with tf.name_scope("dnn1"):
        with tf.device("/job:local/task:0"):
            first_batch = tf.slice(x, [0,0], [SPLITNUM, 0])
            hidlay1 = tf.layers.dense(x, N_LAY1, name="hidlay1", activation=tf.nn.relu)
            hidlay2 = tf.layers.dense(hidlay1, N_LAY2, name="hidlay2", activation=tf.nn.relu)
            hidlay3 = tf.layers.dense(hidlay2, N_LAY3, name="hidlay3", activation=tf.nn.relu)
            outputs1 = tf.layers.dense(hidlay3, NUM_CLASSES, name="outputs1")
    with tf.name_scope("dnn2"):
        with tf.device("/job:local/task:1"):
            second_batch = tf.slice(x, [SPLITNUM, 0], [-1,-1])
            hidlay4 = tf.layers.dense(x, N_LAY4, name="hidlay4", activation=tf.nn.relu)
            hidlay5 = tf.layers.dense(hidlay4, N_LAY5, name="hidlay5", activation=tf.nn.relu)
            hidlay6 = tf.layers.dense(hidlay5, N_LAY6, name="hidlay6", activation=tf.nn.relu)
            outputs2 = tf.layers.dense(hidlay6, NUM_CLASSES, name="outputs2")
            outputs=(outputs1+outputs2)/2
'''
