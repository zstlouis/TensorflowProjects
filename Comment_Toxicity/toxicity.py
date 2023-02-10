import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, Embedding, LSTM


# load data into a dataframe
df = pd.read_csv('train.csv')
print(df.head())
# view all columns in dataframe
print(df.columns)

# view example of a comment that we will be processing
print(df.iloc[0]['comment_text'])

# use TextVectorization to convert text to an integer sequence
# ['I love you'] -> [3,54,1]

# separate comment text to process and the columns that will be used to classify text as toxic or not
# [toxic severe_toxic, obscene, threat, insult, identity_hate]
x = df['comment_text']
print(x)
y = df[df.columns[2:]].values
print(y)

# setup TextVectorization
# number of words in available vocab
MAX_WORDS = 200000
vectorizer = TextVectorization(max_tokens=MAX_WORDS, output_sequence_length=1800, output_mode='int')

vectorizer.adapt(x.values)

vectorized_text = vectorizer(x.values)

print(vectorized_text)

# create data pipeline
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(16000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8) # helps prevents bottlenecks

# view a batch from the dataset
# can then separate feature and labeled data by assigning to x and y variables
# batch_x, batch_y = dataset.as_numpy_iterator().next()
print(dataset.as_numpy_iterator().next())

# split data into train/val/test sets
# .take -> extract batch size from data
# .skip -> ignore batch size from data
train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))
print(len(train))
print(len(val))
print(len(test))

# create the model
model = Sequential([
    # 32 features in embedding
    Embedding(MAX_WORDS+1, 32),
    Bidirectional(LSTM(32)),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(6, activation='sigmoid')
])

print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])
history = model.fit(train, epochs=1, validation_data=val)

# make predictions
