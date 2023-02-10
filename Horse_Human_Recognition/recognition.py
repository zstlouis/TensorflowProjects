import os
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# create callback
class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.75):
            print('\nAccuracy reached 98%. Exiting training')
            self.model.stop_training = True


local_zip = 'archive.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('horse-or-human')
zip_ref.close()

train_horse_dir = os.path.join('horse-or-human/horse-or-human/train/horses')
train_human_dir = os.path.join('horse-or-human/horse-or-human/train/humans')

validation_horse_dir = os.path.join('horse-or-human/horse-or-human/validation/horses')
validation_human_dir = os.path.join('horse-or-human/horse-or-human/validation/humans')

# list the first 10 images in both horse and humans train directories
train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_humans_names = os.listdir(train_human_dir)
print(train_humans_names[:10])

# store validation images
validation_horse_names = os.listdir(validation_horse_dir)
validation_humans_names = os.listdir(validation_human_dir)

# get total number of both horse and human images
print(len(os.listdir(train_horse_dir)))
print(len(os.listdir(train_human_dir)))
print(len(os.listdir(validation_horse_dir)))
print(len(os.listdir(validation_human_dir)))

# plot a few images to get a better idea of what the images look like
# parameters for graph
# will plot images 4x4

# nrows, ncols = 4, 4
# pic_idx = 0
#
# fig = plt.gcf()
# fig.set_size_inches(ncols * 4, ncols * 4)
# pic_idx += 8
#
# next_horse_pic = [os.path.join(train_horse_dir, fname) for fname in train_horse_names[pic_idx-8: pic_idx]]
# next_human_pic = [os.path.join(train_human_dir, fname) for fname in train_humans_names[pic_idx-8: pic_idx]]
#
# for i, img_path in enumerate(next_horse_pic+next_human_pic):
#     sp = plt.subplot(nrows, ncols, i + 1)
#
#     img = mpimg.imread(img_path)
#     plt.imshow(img)
#
# plt.show()


# create model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), input_shape=(300, 300, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# set up data augmentation for better model training on images
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    'horse-or-human/horse-or-human/train',
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary'
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_directory(
    'horse-or-human/horse-or-human/validation',
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary'
)
callBack = myCallBack()

# steps size -> amount of items in dataset / by batch_size
#                       1027/128 = ~8
model.fit(train_generator, validation_data=validation_generator, epochs=10, callbacks=[callBack], steps_per_epoch=8,
          validation_steps=8, verbose=1)
