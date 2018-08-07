import tensorflow_hub as hub
import keras.layers as layers
from keras.models import Model
from keras.layers import Dense, Input
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data/raw/mnist')
mnist_hub_dir = '../data/hub_module/mnist_module'
keras_model_weight_dir = '../data/keras_model/include_tensorflowhub_model_weight.h5'


def batch_iter(batch_size, phase="train"):

    if phase == "train":
        num_data = mnist.train.num_examples
    if phase == "test":
        num_data = mnist.test.num_examples
    batch_per_epoch = int(num_data / batch_size) + 1

    def data_generator():
        while True:
            for batch_num in range(batch_per_epoch):
                pull_data_length = batch_size
                if batch_size*(batch_num+1) > num_data:
                    pull_data_length = num_data - int(batch_size*batch_num)

                if phase == "train":
                    x, y = mnist.train.next_batch(pull_data_length)
                if phase == "test":
                    x, y = mnist.test.next_batch(pull_data_length)

                yield x, y

    return batch_per_epoch, data_generator()


def train_save_model():

    module = hub.Module(mnist_hub_dir, trainable=True)

    def mnist_hub_fn(x):
        softmax = module(x)
        return softmax

    # Input Layers
    input_img = Input(shape=(784, ), dtype='float32')

    # Hidden Layers
    hidden = layers.Lambda(mnist_hub_fn, output_shape=[10])(input_img)
    hidden = Dense(units=256, activation='relu')(hidden)
    predict = Dense(units=10, activation='softmax')(hidden)

    model = Model(inputs=[input_img], outputs=predict)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    batch_size = 32
    train_steps, train_batch_iter = batch_iter(batch_size, phase="train")
    test_steps, test_batch_iter = batch_iter(batch_size, phase="test")

    model.fit_generator(train_batch_iter, train_steps, epochs=1, validation_data=test_batch_iter, validation_steps=test_steps)

    score = model.evaluate(mnist.test.images, mnist.test.labels)
    print("------ test LOSS and accuracy ------")
    print(score)

    # save keras layers weights without hub module's weights.
    model.save_weights(keras_model_weight_dir)
    print("Saved model to disk")
    # print(model.get_weights())


def load_weight_evaluate_model():

    module = hub.Module(mnist_hub_dir, trainable=True)
    def mnist_hub_fn(x):
        softmax = module(x)
        return softmax

    # Input Layers
    input_img = Input(shape=(784, ), dtype='float32')

    # Hidden Layers
    hidden = layers.Lambda(mnist_hub_fn, output_shape=[10])(input_img)
    hidden = Dense(units=256, activation='relu')(hidden)
    predict = Dense(units=10, activation='softmax')(hidden)

    model = Model(inputs=[input_img], outputs=predict)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    model.load_weights(keras_model_weight_dir)
    score = model.evaluate(mnist.test.images, mnist.test.labels)
    print("------ Load model LOSS and accuracy ------")
    print(score)
    # print(model.get_weights())



if __name__ == '__main__':
    train_save_model()
    load_weight_evaluate_model()


