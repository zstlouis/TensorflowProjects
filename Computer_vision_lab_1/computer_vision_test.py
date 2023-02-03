import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Create callback function.
# Function will exit training once accuracy has reached over 87%
class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.87):
            print('\nReached 87% accuracy so cancelling training')
            self.model.stop_training = True

# load the fashion dataset
mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()

# plot the first image to show that the dataset has been loaded
# plt.figure(figsize=(5,5))
# plt.imshow(training_images[0])
# plt.show()
# print(training_labels[0])

# normalize the data so values are between 0-1 rather then 0-255
training_images = training_images / 255.0
testing_images = testing_images / 255.0

# create callback object
callbacks = myCallBack()

# create model to process images
# when dealing with images you first need to flatten images to 1 dimension
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# training the model on training set for 5 epochs
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

# test on new image data
model.evaluate(testing_images, testing_labels)