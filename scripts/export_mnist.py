import argparse
import sys
import tensorflow as tf
import tensorflow_hub as hub
from keras.models import load_model

IMAGE_VAR_NAME = "image_tensor"

def train_mnist():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train[:10], y_train[:10], epochs=1)
    model.evaluate(x_test, y_test)

    return model


def raw_tensorflow():
    # データのインポート
    from tensorflow.examples.tutorials.mnist import input_data
    import tensorflow as tf

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    sess = tf.InteractiveSession()

    # モデルの作成
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # 損失とオプティマイザーを定義
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # 訓練
    tf.initialize_all_variables().run()
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        train_step.run({x: batch_xs, y_: batch_ys})

    # 訓練モデルのテスト
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))




def make_module_spec():

    def module_fn():
        # mnist = tf.keras.datasets.mnist
        #
        # (x_train, y_train),(x_test, y_test) = mnist.load_data()
        # x_train, x_test = x_train / 255.0, x_test / 255.0
        #
        # model = tf.keras.models.Sequential([
        #   tf.keras.layers.Flatten(),
        #   tf.keras.layers.Dense(512, activation=tf.nn.relu),
        #   tf.keras.layers.Dropout(0.2),
        #   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        # ])
        # model.compile(optimizer='adam',
        #               loss='sparse_categorical_crossentropy',
        #               metrics=['accuracy'])
        #
        # model.fit(x_train[:10], y_train[:10], epochs=1)
        # model.evaluate(x_test, y_test)
        #
        # input_image = tf.placeholder(name="input_image", shape=[1, 28, 28], dtype=tf.float32)


        from tensorflow.examples.tutorials.mnist import input_data

        mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data')

        # Create the model
        x = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.matmul(x, W) + b

        # Define loss and optimizer
        y_ = tf.placeholder(tf.int64, [None])


        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        # Train
        for _ in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(
            accuracy, feed_dict={
                x: mnist.test.images,
                y_: mnist.test.labels
            }))

        predict = tf.


        hub.add_signature("default", {"input_image": x},
                      {"default": predict_vector})

    return hub.create_module_spec(module_fn)


def export(export_path):
    # Write temporary vocab file for module construction.
    spec = make_module_spec()

    with tf.Graph().as_default():
        m = hub.Module(spec)
        # The embeddings may be very large (e.g., larger than the 2GB serialized
        # Tensor limit).  To avoid having them frozen as constant Tensors in the
        # graph we instead assign them through the placeholders and feed_dict
        # mechanism.

        # p_image = tf.placeholder(tf.float32)
        # image_predicting = tf.assign(m.variable_map[IMAGE_VAR_NAME], p_image)

        with tf.Session() as sess:
#            sess.run(image_predicting, feed_dict={p_image: image})
            sess.run(m)
            m.export(export_path, sess)


def main(_):
    export(FLAGS.export_path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--export_path",
      type=str,
      default=None,
      help="Where to export the module.")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)