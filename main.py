import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

data = tf.keras.datasets.fashion_mnist # Accesses the fashion_mnist dataset in the keras library

(training_images, training_labels), (test_images, test_labels) = data.load_data()
# Fills the first tuple with 60,000 training images and the second tuple with 10,000 testing images.

training_images = training_images / 255.0
test_images = test_images / 255.0
# Divides every value in the arrays by 255.0, reducing them to a value between 0 and 1. This "normalizes" the arrays.
# For, uh, reasons...normalized data leads to better performance when training neural networks. Often, when your data is
# not normalized, your network will not learn and will have massive errors.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    # Input layer specification: our inputs are 28x28 images, but we want them to be treated as a series of numeric
    # values. Flatten takes that "square" value (a 2D array) and turns it into a line (a 1D array).

    keras.layers.Dense(128, activation=tf.nn.relu),
    # A layer of neurons: we want 128 of them. This number is entirely arbitrary: as you design the layers you will want
    # to pick the appropriate number of values to enable your model to actually learn. More neurons introduces problems:
    # 1.) More neurons means the code will run slower
    # 2.) More neurons can lead to "overfitting", where the NN is only accurate for the training data.
    # On the other hand, less neurons may mean that the model might not have sufficient parameters to learn.

    # This layer also specifies an activation function (code that will execute on each neuron in the layer). TensorFlow
    # supports many of them, but a very common one in the middle layers is relu, which stands for rectified linear unit,
    # a simple function that returns a value if it's greater than 0.

    keras.layers.Dense(10, activation=tf.nn.softmax)
    # Output layer: this has 10 neurons, because we have 10 classes. Each of these neurons will end up with a
    # probability that the input pixels match that class, so our job is to determine which one has the highest value.
    # The softmax function picks the highest value for us.
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Sparse categorical cross entropy is a categorical loss function. Since the output of our NN is a category (belongs to
# 1 of 10 categories of clothing), SCCE is the way to go.
# The adam optimizer is a faster and more efficient variant of stochastic gradient descent.
# The accuracy metric reports the accuracy of the neural network during training and evaluation.

model.fit(training_images, training_labels, epochs=5)
# The network is trained over five epochs by fitting the training images over the training labels.

model.evaluate(test_images, test_labels)
# The network is evaluated using 10,000 images and labels for testing. This one line of code passes said images to the
# trained model to have it predict what it thinks each image is, compare that to its actual label, and sum up the
# results.

classifications = model.predict(test_images)
# Stores a set of classifications
print(classifications[0])
# Prints the values of the 10 neurons for the first classification.
print(test_labels[0])
# Prints the test label of the first test image.
