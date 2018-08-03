import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data/raw/mnist')

mnist_model_export_dir = '../data/model/mnist_model'
mnist_hub_dir = '../data/hub_module/mnist_module'


# create a simple saved_model
def generate_freeze_saved_mnist_model(export_dir):

    with tf.Graph().as_default() as graph:

        # create model
        x = tf.placeholder(tf.float32, [None, 784], name="x")
        W = tf.get_variable(initializer=tf.zeros([784, 10]), name="w")
        b = tf.get_variable(initializer=tf.zeros([10]), name="b")
        y = tf.add(tf.matmul(x, W), b, name="y")
        y_ = tf.placeholder(tf.int64, [None])

        # define loss and optimizer
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # train
            for _ in tqdm(range(1)):
                batch_xs, batch_ys = mnist.train.next_batch(100)
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            # test trained model
            correct_prediction = tf.equal(tf.argmax(y, 1), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("------ trained model accuracy ------")
            print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

            # save model
            builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
            builder.add_meta_graph_and_variables(sess, "")
            builder.add_meta_graph("", strip_default_attrs=True)
            builder.save()


def get_mnist_module_fn(model_export_dir):
    variable_dict = {}

    # define module function for TensorFlow hub module spec
    def module_fn():
        graph = tf.get_default_graph()
        with tf.Session(graph=graph) as sess:
            tf.saved_model.loader.load(sess, "", model_export_dir)

        x = graph.get_tensor_by_name('x:0')
        print(x)
        y = graph.get_tensor_by_name('y:0')
        print(y)

        hub.add_signature('default', inputs={'x': x}, outputs={'default': y})

    # save variable from trained model.
    # convert variable Tensor to numpy array.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.saved_model.loader.load(sess, "", model_export_dir)

        for v in tf.trainable_variables():
            name = v.value().name
            value = v.eval()
            variable_dict[name] = value

    return module_fn, variable_dict


def export_hub_from_seved_model(model_export_dir, hub_dir):

    # get module function and variable array dict
    module_fn, variable_dict = get_mnist_module_fn(model_export_dir)
    spec = hub.create_module_spec(module_fn=module_fn, drop_collections=['losses', 'train_op'])

    with tf.Graph().as_default():
        m = hub.Module(spec)

        print("------ check variables ------")
        print(tf.global_variables())
        print(variable_dict)

        # assign variable to graph
        init_w = tf.assign(m.variable_map["w"], tf.convert_to_tensor(variable_dict["w/read:0"]))
        init_b = tf.assign(m.variable_map["b"], tf.convert_to_tensor(variable_dict["b/read:0"]))
        init_vars = tf.group([init_w, init_b])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run([init_vars])

            # export TensorFlow hub module
            m.export(hub_dir, sess)


def test_and_retraining_hub_module(hub_dir):

    # test hub module
    with tf.Graph().as_default():

        # load hub module
        module = hub.Module(hub_dir, trainable=True)

        # create model
        x = tf.placeholder(tf.float32, [None, 784], name="x")
        y_ = tf.placeholder(tf.int64, [None], name="y_")
        y = module(x)

        # define loss and optimizer
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # check accuracy
            correct_prediction = tf.equal(tf.argmax(module(x), 1), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # test loaded hub module
            print("------ test load module accuracy ------")
            print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

            # retrain hub module
            print("------ retrained module ------")
            for _ in tqdm(range(1)):
                batch_xs, batch_ys = mnist.train.next_batch(100)
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            # test retrained hub module
            print("------ test retrained module accuracy ------")
            print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    generate_freeze_saved_mnist_model(mnist_model_export_dir)
    export_hub_from_seved_model(mnist_model_export_dir, mnist_hub_dir)
    test_and_retraining_hub_module(mnist_hub_dir)