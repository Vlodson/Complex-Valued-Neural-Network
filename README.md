# Complex Valued Deep Neural Network Module

This module provides a Keras-like API for creating complex valued deep neural networks.

Complex valued neural networks work much the same way as real valued deep neural networks, except for the fact that all the weights, biases and neuron values are complex numbers. The module utilizes fully complex valued activations and losses to train models on complex valued datasets.

Within the module are all the implementations for complex valued dense layers, losses, activations and a model class to build models from scratch. Each part is modular and easily expandable.

# Modeling CVNNs

This section will provide a guide on how to create, train and test a CVNN model.

## Creating a CVNN Layer

To create a CVNN layer use the Layer class from the layers package.

Creating a single fully connected layer is as simple as initializing a Layer object, which can be done with the following snippet

```python
from layers import Layer

layer = Layer(neurons=3, activation="linsca")
```

The init method for a layer takes the number of neurons that the layer will have, an activation function that the layer will use and optionally the shape of the input to the layer. The last argument is needed only for the first layer of the network, but it can be passed to other layers as well.

The activation function can be passed by its name or by creating the activation function object. Names of available activations are available in *activations/activations_dict.py*. Each activation function available is presented more in depth in the paper.

## Creating a CVNN Model

To create a model use the Model class from the model package. Each model is comprised of a list of layers, an optimizer, a loss function and model metrics.

The first step is to create a Model object, which can be done with the following snippet

```python
from model.model import Model
from layers.layer import Layer

model = Model(
  [
    Layer(neurons=5, activation="linsca", input_shape=10),
    Layer(neurons=3, activation="linsca"),
    Layer(neurons=1, activation="linsca"),
  ]
)
```

The first layer has to have the input_shape parameter so that the model knows how to initialize the layer parameters, for all other layers it is optional.

The last layer has to have a single neuron due to the classification method that is used. More on this can be found in the paper.

The next step is to build the model, or in other words, to set the model's optimizer, loss and metrics and for the model to initialize all the parameters it needs to work. For this purpose, the Model class provides the build method.

```python
model.build(
  optimizer="adam",
  loss="mse",
  metrics=["accuracy", "f1"],
)
```

Same as with activations, all the parameters of Model.build can be passed either as their respective class instances or as their names represented as strings. All the names and their initializations are present in their respective *_dict.py* files.

## Training a CVNN Model

After building the model, it is time to train the model on a dataset. The CVNN Module provides two functions for creating synthetic datasets which can be found in the *utils/synthetic_data.py*. Datsets created this way come normalized and shuffled by default.

To train a model, the user needs to pass the training data and labels, the batch size, the number of epochs of trainig and validation data and labels.

```python
model.train(
  x=x_train,
  y=y_train,
  batch_size=64,
  epochs=512,
  x_val=x_val,
  y_val=y_val
)
```

During training, a progress bar will show the current epoch, how much time has elapsed, how much time is left for training to be finished and the values for training and validation loss.

Loss and metrics values are stored within the History class, which is a public field within the model itself. The history class saves all the states of the model's metrics and loss throughout the epochs of training. This data can then be accessed or plotted using methods provided within the History class.

## Testing a CVNN Model

Testing the model is as simple as calling the *test* method of an already built, and ideally trained, model. The method requires two arguments, the test data and test labels.

```python
model.test(x_test, y_test)
```

Upon finishing the method will return a dictionary with the values for loss and all the passed metrics.

## Examples

A full example of building a dataset; creating, building, training and testing a model are available inside *scripts.model_script.py*

## Hardware Optimizations

Both GPU and CPU optimizations are implemented for all operations needed to run a model, by wrapping the required numpy functions in either Cupy or decorating numpy funcitons using Numba.

By default Cupy will try to be imported. But in the absence of Cupy, the numpy wrapper will then import the Numba wrapper. There is no fallback if neither are available.

# User Defined Classes

If you wish to create your own activation, loss, metric or optimizer, you can do so by inheriting the base class of whatever you want to create and implementing the methods that are required. This section will shortly gloss over each of the base classes and their required methods.

## Activations

To create a custom activation, first inherit the activation abstract class, then implement the activate and deactivate methods. The activate method applies the activation function to the passed input, while the deactivate function applies the derivate of the activation function to the passed input.

## Losses

For a custom loss function, inherit the Loss base class and implement calculate loss and loss gradient methods. The calculate loss method applies the loss function to a set of labels and their respective predictions. The loss gradient method applies the derivate of the loss function with respect to the predictions given a set of labels and predictions.

## Metrics

For metrics, after inheriting the Metric class, implement the calculate metric method that applies the metric's function to a set of labels and predictions.

## Optimizers

For optimizers, first after inheriting the Optimizer base class, set any extra parameters that the optimizer might need in the init method. These might include parameters such as learn rate or epsilon normalization.

Then implement the build method, which initializes any parameters that couldn't be initialized during the init method and/or require information that is stored in layers. As an example, the initialization of previous iteration's gradients can be done during the build state of the optimizer.

Lastly implement the update parameters method that updates the parameters using the optimizers chosen optimization method.
