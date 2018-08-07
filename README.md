# TensorFlow Hub experiments
[TensorFlow hub](https://www.tensorflow.org/hub/) is pretrained model for TensorFlow.
This repository makes the model to solve MNIST and export model to TensorFlow Hub module.
Following experiments are included in this repository.

* Trained model and export the hub module, and use it.
* Fine-tuning hub bodule, and save model inculuded hub module.
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
