import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import keras.layers as layers
from keras.models import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Input
from tensorflow.examples.tutorials.mnist import input_data
from keras.engine.topology import Layer
from keras import backend as K

mnist = input_data.read_data_sets('../data/raw/mnist')
mnist_hub_dir = '../data/hub_module/mnist_module'




class TrainableHubRayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(TrainableHubRayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(TrainableHubRayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)





def batch_iter(batch_size, phase="train"):

    if phase == "train":
        num_data = mnist.train.num_examples
    if phase == "test":
        num_data = mnist.test.num_examples
    batch_per_epoch = int(num_data / batch_size) + 1

    print(batch_per_epoch)

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
    predict = layers.Lambda(mnist_hub_fn, output_shape=[10])(input_img)
    # hidden = Dense(units=256, activation='relu')(hidden)
    # predict = Dense(units=10, activation='softmax')(hidden)

    model = Model(inputs=[input_img], outputs=predict)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    print(mnist.train.images.shape)

    batch_size = 32
    train_steps, train_batch_iter = batch_iter(batch_size, phase="train")
    test_steps, test_batch_iter = batch_iter(batch_size, phase="test")

    model.fit_generator(train_batch_iter, train_steps, epochs=1, validation_data=test_batch_iter, validation_steps=test_steps)

    score = model.evaluate(mnist.test.images, mnist.test.labels)
    print(score)

    # Todo Saved model with retrained module of tensorflow hub
    # model.save("../data/keras_model/mnist_keras_model.h5")

    # serialize model to JSON
    # model_json = model.to_json()
    # with open("../data/keras_model/include_tensorflowhub_model.json", "w") as json_file:
    #     json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("../data/keras_model/include_tensorflowhub_model_weight.h5")
    print("Saved model to disk")

    print(model.get_weights())


def load_evaluate_model():
    model = load_model("../data/keras_model/mnist_keras_model.h5")
    model.add(Dense(513, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.fit(mnist.train.images, mnist.train.labels, epochs=1)

    score = model.evaluate(mnist.test.images, mnist.test.labels)
    print(score)

    batch_xs, batch_ys = mnist.train.next_batch(1)
    print(model.predict(batch_xs))
    print(batch_ys)


def load_weight_evaluate_model():
    model_file = "../data/keras_model/include_tensorflowhub_model_weight.h5"

    module = hub.Module(mnist_hub_dir, trainable=True)
    def mnist_hub_fn(x):
        softmax = module(x)
        return softmax

    # Input Layers
    input_img = Input(shape=(784, ), dtype='float32')

    # Hidden Layers
    predict = layers.Lambda(mnist_hub_fn, output_shape=[10])(input_img)
    # hidden = Dense(units=256, activation='relu')(hidden)
    # predict = Dense(units=10, activation='softmax')(hidden)

    model = Model(inputs=[input_img], outputs=predict)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    model.load_weights(model_file)
    score = model.evaluate(mnist.test.images, mnist.test.labels)
    print(score)

    print(model.get_weights())



if __name__ == '__main__':
    train_save_model()
    # load_evaluate_model()
    load_weight_evaluate_model()


