import tensorflow as tf
import numpy as numpy
import matplotlib.pyplot as plt


# create callback to exit training once accuracy has reached 85%
class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > .98):
            print('\n 98% accuracy reached. Exiting training')
            self.model.stop_training = True


# load fashion dataset
mnist = tf.keras.datasets.fashion_mnist

# split into training and testing data
(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()

# reshape images to add dimensionality layer (black and white images so dimensionality of 1
training_images = training_images.reshape(60000, 28,28,1)
testing_images = testing_images.reshape(10000, 28,28,1)

# normalize the data
training_images = training_images / 255.0
testing_images = testing_images / 255.0

# plot the first image to show dataset has been loaded
# plt.imshow(training_images[0])
# plt.show()

# initialize callback
callback = myCallBack()


# create the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), input_shape=(28,28, 1), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,2), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=20, callbacks=[callback])




