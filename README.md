# TensorFlow Hub experiments
[TensorFlow hub](https://www.tensorflow.org/hub/) is pretrained model for TensorFlow.
This repository makes the model to solve MNIST and export model to TensorFlow Hub module.
Following experiments are included in this repository.

* Trained model and export the hub module, and use it.
* Fine-tuning hub module, and save model inculuded hub module.
* Using own hub module in Keras.


## Directory layout
    .
    ├── data                
    │   ├── hub_module      # Hub module exported
    │   ├── model           # Model of solving MNIST
    │   └── raw             # MNIST dataset
    └── scripts             # Experimtens codes


# Get started
```
$ pip install -r requirements.txt
$ cd scripts
```

# Experiments

## 1. Make and export MNIST model
Make model and train with the training dataset of 500 records against MNIST. (This training dataset is "VERY SMALL" because observe efficient of fine-tuning.) In this case, trained model's accuracy is "0.681".

Next, load making hub module and fine-tuning that with another 500 records. Retrained model's accuracy "0.8209" outperform the before model.

```
$ python export_hub_for_solving_mnist.py
100%|████████████████████████████████████████████████| 5/5 [00:00<00:00, 24.27it/s]
------ trained model accuracy ------
0.681
Tensor("x:0", shape=(?, 784), dtype=float32)
Tensor("y:0", shape=(?, 10), dtype=float32)
------ check variables ------
[<tf.Variable 'module/w:0' shape=(784, 10) dtype=float32_ref>, <tf.Variable 'module/b:0' shape=(10,) dtype=float32_ref>]
{'w/read:0': array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]], dtype=float32), 'b/read:0': array([-0.04940301,  0.05061443, -0.01812078, -0.00535163,  0.04177742,
        0.02548235, -0.01509129,  0.01740718, -0.01047494, -0.03683973],
      dtype=float32)}
------ test load module accuracy ------
0.681
------ retrained module ------
100%|███████████████████████████████████████████████| 5/5 [00:00<00:00, 112.00it/s]
------ test retrained module accuracy ------
0.8209
```


## 2. Using hub module and adding the graph, save the new model.
Load hub module and add new graph. Then, train new model (add graph is trained, hub module is fine-tuned) and evaluate new model. As a result, new model accuracy is "0.78".

```
$ python using_hub_example.py
------ test only hub module accuracy ------
0.59
100%|█████████████████████████████████████████████████| 5/5 [00:00<00:00, 20.60it/s]
------ trained new model with hub module accuracy ------
0.75
------ loading new model with hub module accuracy ------
0.78
```

In addition, I introduce hosting hub module by your cloud storage. 
First, you compress to tar ball with the following command.(Linux and Unix commands)

```
$ tar cfz mnist_hub.tgz ../data/hub_module/mnist_module/
```

Second, upload tgz to your cloud storage like AWS S3, Google Cloud Storage.

Finally, download hub module from your storage with the additional query.

```
mnist_hub_dir = 'https://your_hosting_address/mnist_module.tgz?tf-hub-format=compressed'
```


## 3. Using hub module with keras
Using hub module with keras's sequential layer. TensorFlow hub is used in Lambda Layer and add two Dence layers. Train and test this model and save keras layer's weithts. After, Confirm seved modle's performance by make same model and load weights and evaluate accuracy.

```
$ python using_hub_with_keras_example.py 
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 784)               0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 10)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               2816      
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 5,386
Trainable params: 5,386
Non-trainable params: 0
_________________________________________________________________
Epoch 1/1
2018-08-07 16:43:55.607675: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
1719/1719 [==============================] - 4s 2ms/step - loss: 0.5101 - acc: 0.8409 - val_loss: 0.3817 - val_acc: 0.8772
10000/10000 [==============================] - 0s 22us/step
------ test LOSS and accuracy ------
[0.3817396643638611, 0.8772]
Saved model to disk
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 784)               0         
_________________________________________________________________
lambda_2 (Lambda)            (None, 10)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 256)               2816      
_________________________________________________________________
dense_4 (Dense)              (None, 10)                2570      
=================================================================
Total params: 5,386
Trainable params: 5,386
Non-trainable params: 0
_________________________________________________________________
10000/10000 [==============================] - 0s 33us/step
------ Load model LOSS and accuracy ------
[0.3817396643638611, 0.8772]

```

# future work
* Save Keras model with hub module weights. Now, there have the problem of keras couldn't save Lambda Layer weight. One way, convert Lambda Layer to Custom Layer that can be saved weights.


