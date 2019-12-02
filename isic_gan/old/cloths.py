# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Load the images and labels (train and test)
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Load the labels names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Print some info about the data structres of images and labels
# print(train_images.shape)
# print(train_labels.shape)
# print(train_labels)
# print(test_images.shape)
# print(test_labels.shape)
# print(test_labels)

# Plot the first image on train set
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# Adjuste the pixels values of the images from [0,255] to [0,1]
train_images = train_images/255.0
test_images = test_images/255.0

# Plot the first image on train set again
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# Plot the first 25 images from the training set to check the labels
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# Setup model
# 1st Layer: Flatten layers transforms the 2d-arrays of images (28x28 pixels)
# into a 1d-array of 28*28 = 784 pixels, has no paramters to learn, just reformats the data.
# Second layer: Fully connected with 128 nodes (or neurons).
# Third layer: The 10 neurons with softmax represents an array of 10 probabilities
# scores that sum to 1. Each node corresponds to the probability of the input
# belong to one of the 10 classes.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # Input layer same size as images
    keras.layers.Dense(128, activation=tf.nn.relu), # Second layer, is a fully connected layer with 128 neurons using the RELU activation function
    keras.layers.Dense(10, activation=tf.nn.softmax) # Last layers is a fully connected layer with 10 neurons using the Softmax activation function (Same number of neurons as are classes)
])

# Compile the model
# Loss function: Mesures how accurate the model is, the goal is to minimize this
# Optimizer: Is how the model is updated based on the data it sees and its loss function
# Metrics: Used to monitor the training and testis steps. Here we use the accuracy
model.compile(optimizer=tf.train.AdamOptimizer(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Train the model
# 1. Feed the training data to the model (train_images and train_labels)
# 2. The model learns to associate images and labels
# 3. We as the model to make predictions about a test set (test_images).
# We verify that the predictions match the labels from the test_labels array.

# Train
model.fit(train_images, train_labels, epochs=10)

# Test
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Make predictions
# With the model trained, we can use it to make predictions about some images
predictions = model.predict(test_images) # Each array has 10 numbers representing the confidence (probability) of the input be each class
# print(predictions[0])

# print( 'Correct? %s' % (np.argmax(predictions[0]) == test_labels[0]) ) # Get higher probability position and compare with the correct

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                            100*np.max(predictions_array),
                                            class_names[true_label]),
                                            color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Plot the results for image 12
# i = 12
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions, test_labels)

# Plot the first X test images, their predicted label, and the true label
# Color correct prediction in blue, incorrect predictions in red
# num_rows = 8
# num_cols = 4
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2*num_cols, 2*i+1)
#     plot_image(i, predictions, test_labels, test_images)
#     plt.subplot(num_rows, 2*num_cols, 2*i+2)
#     plot_value_array(i, predictions, test_labels)
#
# plt.show()

# Test a single image
# Grab an image from the test dataset and adjust the dimensions to a array of images
img = (np.expand_dims(test_images[0], 0))
predictions_single = model.predict(img)
print(test_labels[0])
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
