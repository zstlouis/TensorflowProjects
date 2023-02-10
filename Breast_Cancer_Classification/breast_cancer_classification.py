import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# load dataset
breast_cancer_dataframe = pd.read_csv('data.csv')
print(breast_cancer_dataframe.head())

# drop last column in dataframe
breast_cancer_dataframe.drop(columns=breast_cancer_dataframe.columns[-1], axis=1, inplace = True)
print(breast_cancer_dataframe.head())

# convert diagnosis column from string to int values
# M -> 0
# B -> 1

breast_cancer_dataframe['diagnosis'] = breast_cancer_dataframe['diagnosis'].map({'M': 0, 'B': 1})
print(breast_cancer_dataframe.tail())

# view shape of dataset and if any data is missing
print(breast_cancer_dataframe.shape)
print(breast_cancer_dataframe.isna().sum())

# view distribution of diagnosis column
print(breast_cancer_dataframe['diagnosis'].value_counts())

# view averages of each column for each classification type
print(breast_cancer_dataframe.groupby('diagnosis').mean())

# separate into features data and target label data
x = breast_cancer_dataframe.drop(columns='diagnosis', axis=1)
y = breast_cancer_dataframe['diagnosis']
print(x.head())
print(y.head())

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
print(X_train.shape)
print(y_train.shape)

# standardize the dataset
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

print(X_train_std)


# create NN model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(31,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_std, y_train, validation_split=0.2, epochs=10)

# plot the model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['training data', 'validation data'])
plt.show()

# plot the model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['training data', 'validation data'])
plt.show()


