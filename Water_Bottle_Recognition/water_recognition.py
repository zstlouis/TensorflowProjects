import os
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import scipy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_full_water_dir = os.path.join('WaterBottles/train/Full_Water_level/Full _Water_level')
train_full_water_files = os.listdir(train_full_water_dir)

train_half_water_dir = os.path.join('WaterBottles/train/Half_water_level/Half_water_level')
train_half_water_files = os.listdir(train_half_water_dir)

train_overflowing_water_dir = os.path.join('WaterBottles/train/Overflowing/Overflowing')
train_overflowing_water_files = os.listdir(train_overflowing_water_dir)

validation_full_water_dir = os.path.join('WaterBottles/validation/Full_Water_level/Full _Water_level')
validation_full_water_files = os.listdir(validation_full_water_dir)

validation_half_water_dir = os.path.join('WaterBottles/validation/Half_water_level/Half_water_level')
validation_half_water_files = os.listdir(validation_half_water_dir)

validation_overflowing_water_dir = os.path.join('WaterBottles/validation/Overflowing/Overflowing')
validation_overflowing_water_files = os.listdir(validation_overflowing_water_dir)

# split full_water_files to train and test
# FULL_WATER_SPLIT_POINT = int(len(full_water_files) * .9)
# HALF_WATER_SPLIT_POINT = int(len(half_water_files) * .9)
# OVERFLOWING_WATER_SPLIT_POINT = int(len(overflowing_water_files) * .9)

# print(len(full_water_files[:SPLIT_POINT]))
# print(len(full_water_files))
# print(len(full_water_files[SPLIT_POINT:]))

def resize(path):
    f = path
    for file in os.listdir(f):
        f_img = f + "/" + file
        img = Image.open(f_img)
        img = img.resize((300, 300))
        img.save(f_img)


resize(train_full_water_dir)
resize(train_half_water_dir)
resize(train_overflowing_water_dir)

resize(validation_full_water_dir)
resize(validation_half_water_dir)
resize(validation_overflowing_water_dir)


#
# nrows, ncols = 4, 4
# pic_idx = 0
#
# fig = plt.gcf()
# fig.set_size_inches(ncols * 4, ncols * 4)
# pic_idx += 8
#
# next_full_water_pic = [os.path.join(train_full_water_dir, fname) for fname in
#                        train_full_water_files[pic_idx - 8: pic_idx]]
#
# for i, img_path in enumerate(next_full_water_pic):
#     sp = plt.subplot(nrows, ncols, i + 1)
#
#     img = mpimg.imread(img_path)
#     plt.imshow(img)
#
# plt.show()


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), input_shape=(300,300, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=.2,
                                   zoom_range=.2,
                                   rotation_range=.4,
                                   )
train_generator = train_datagen.flow_from_directory(
    'WaterBottles/train',
    target_size=(300,300),
    batch_size=60,
    color_mode='rgb',
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    'WaterBottles/validation',
    target_size=(300,300),
    batch_size=60,
    color_mode='rgb',
    class_mode='categorical'
)

model.fit(train_generator, validation_data=validation_generator, epochs=10, steps_per_epoch=5,
          validation_steps=5, verbose=2)


