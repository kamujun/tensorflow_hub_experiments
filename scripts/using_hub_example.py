import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data/raw/mnist')
mnist_hub_dir = '../data/hub_module/mnist_module'
# mnist_hub_dir = 'https://your_hosting_address/mnist_module.tgz?tf-hub-format=compressed'


def test_and_retraining_hub_module(hub_dir):

    # test hub module
    with tf.Graph().as_default():
        # load hub module
        module = hub.Module(hub_dir, trainable=True, name="y")

        # create model
        x = tf.placeholder(tf.float32, [None, 784], name="x")
        y_ = tf.placeholder(tf.int64, [None], name="y_")
        y = module(x)

        # define loss and optimizer
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        print(x)
        print(y)
        print(y_)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # test hub module
            batch_xs, batch_ys = mnist.test.next_batch(100)

            # Test trained model
            correct_prediction = tf.equal(tf.argmax(module(x), 1), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))

            # retrain hub module
            for _ in tqdm(range(100)):
                batch_xs, batch_ys = mnist.train.next_batch(100)
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            # test retrained hub module
            batch_xs, batch_ys = mnist.test.next_batch(100)
            print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))

            # save model
            builder = tf.saved_model.builder.SavedModelBuilder("../data/model/retrain/")
            builder.add_meta_graph_and_variables(sess, "")
            builder.add_meta_graph("", strip_default_attrs=True)
            builder.save()


    # load and evaluate model by retrained hub module
    with tf.Graph().as_default():

        graph = tf.get_default_graph()
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            tf.saved_model.loader.load(sess, "", "../data/model/retrain/")
            x = graph.get_tensor_by_name('x:0')
            print(x)
            y = graph.get_tensor_by_name('y_apply_default/y:0')
            print(y)
            y_ = graph.get_tensor_by_name('y_:0')
            print(y_)

            # test trained model
            correct_prediction = tf.equal(tf.argmax(y, 1), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            batch_xs, batch_ys = mnist.test.next_batch(100)
            print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))



def add_and_retraining_hub_module(hub_dir):

    # test hub module
    with tf.Graph().as_default():
        # load hub module
        module = hub.Module(hub_dir, trainable=True)

        # create model
        x = tf.placeholder(tf.float32, [None, 784], name="x")
        y_ = tf.placeholder(tf.int64, [None], name="y_")

        y = module(x)
        W2 = tf.get_variable(initializer=tf.zeros([10, 10]), name="w2")
        b2 = tf.get_variable(initializer=tf.zeros([10]), name="b2")
        y2 = tf.add(tf.matmul(module(x), W2), b2, name="y2")

        print(y)
        print(y2)

        # define loss and optimizer
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y2)
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # test hub module
            correct_prediction = tf.equal(tf.argmax(module(x), 1), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # Test trained model
            batch_xs, batch_ys = mnist.test.next_batch(100)
            print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))

            # retrain hub module
            for _ in tqdm(range(100)):
                batch_xs, batch_ys = mnist.train.next_batch(100)
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            # test retrained hub module
            correct_prediction = tf.equal(tf.argmax(y2, 1), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            batch_xs, batch_ys = mnist.test.next_batch(100)
            print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))


if __name__ == '__main__':
    test_and_retraining_hub_module(mnist_hub_dir)
    # add_and_retraining_hub_module(mnist_hub_dir)
