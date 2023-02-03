import os
import tensorflow as tf
import random
import shutil
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = 'lego/star-wars-images/'
names = ["YODA", "LUKE SKYWALKER", "R2-D2", "MACE WINDU", "GENERAL GRIEVOUS"]
tf.random.set_seed(1)

# create train/val/test folders
if not os.path.isdir(BASE_DIR + 'train/'):
    for name in names:
        os.makedirs(BASE_DIR + 'train/' + name)
        os.makedirs(BASE_DIR + 'val/' + name)
        os.makedirs(BASE_DIR + 'test/' + name)

# move images from default location to newly created folders
orig_folders = ["0001/", "0002/", "0003/", "0004/", "0005/"]
for folder_idx, folder in enumerate(orig_folders):
    files = os.listdir(BASE_DIR + folder)
    number_of_images = len([name for name in files])
    n_train = int((number_of_images * 0.6) + 0.5)
    n_valid = int((number_of_images * 0.25) + 0.5)
    n_test = number_of_images - n_train - n_valid
    print(number_of_images, n_train, n_valid, n_test)
    for idx, file in enumerate(files):
        file_name = BASE_DIR + folder + file
        if idx < n_train:
            shutil.move(file_name, BASE_DIR + "train/" + names[folder_idx])
        elif idx < n_train + n_valid:
            shutil.move(file_name, BASE_DIR + "val/" + names[folder_idx])
        else:
            shutil.move(file_name, BASE_DIR + "test/" + names[folder_idx])

# preprocess the data (data augmentation for better training of the data)
# help reduce over fitting in model

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20,
                                   zoom_range=.2)
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'lego/star-wars-images/train',
    target_size=(256, 256),
    class_mode='sparse',
    batch_size=4,
    color_mode='rgb',
    classes=names
)

val_generator = val_datagen.flow_from_directory(
    'lego/star-wars-images/val',
    target_size=(256, 256),
    class_mode='sparse',
    batch_size=4,
    color_mode='rgb',
    classes=names
)

test_generator = test_datagen.flow_from_directory(
    'lego/star-wars-images/test',
    class_mode='sparse',
    batch_size=4,
    color_mode='rgb',
    classes=names
)

# train_batch = train_generator[0]
# print(train_batch[0].shape)
#
# def show(batch, pred_labels=None):
#     plt.figure(figsize=(10,10))
#     for i in range(4):
#         plt.subplot(2,2,i+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(batch[0][i], cmap=plt.cm.binary)
#         lbl = names[int(batch[1][i])]
#         if pred_labels is not None:
#             lbl += "/ Pred:" + names[int(pred_labels[i])]
#         plt.xlabel(lbl)
#     plt.show()
#
# show(train_batch)


# create the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), input_shape=(256, 256, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class myCallBacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.8:
            print('\n Reached 80% accuracy. Exit training.')
            self.model.stop_training = True


myCallBack = myCallBacks()

history = model.fit(train_generator, validation_data=val_generator, epochs=15, callbacks=[myCallBack], verbose=2)


# plot loss and accuracy
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.grid()
plt.legend(fontsize=15)

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.grid()
plt.legend(fontsize=15)

plt.show()


# eval model on new data
model.evaluate(test_generator, verbose=2)


predictions = model.predict(test_generator)
predictions = tf.nn.softmax(predictions)
labels = np.argmax(predictions, axis=1)

print(test_generator[0][1])
print(labels[:4])
