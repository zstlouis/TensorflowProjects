import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
#
# for dirname, _, filenames in os.walk('headlines'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

df = pd.read_json('headlines/Sarcasm_Headlines_Dataset_v2.json', lines=True)
print(df.shape[0])
print(df.columns)

# separate cols and convert to a python list
sentiment = df['headline'].tolist()
labels = df['is_sarcastic'].tolist()
urls = df['article_link'].tolist()
print(sentiment[0], labels[0], urls[0])

# set some hyperparameters
vocab_size = 1000
embed_dim = 16
max_length = 64
SPLIT_POINT = int(df.shape[0] * .9)

# split into training and test sets
train_s = sentiment[:SPLIT_POINT]
train_l = labels[:SPLIT_POINT]
test_s = sentiment[SPLIT_POINT:]
test_l = labels[SPLIT_POINT:]

# convert to numpy arrays
train_sentiment = np.array(train_s)
train_labels = np.array(train_l)
test_sentiment = np.array(test_s)
test_labels = np.array(test_l)


# create tokenizer object to sequence the data
# will then need to add padding to make inputs the same length for processing
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<oov>')
tokenizer.fit_on_texts(train_sentiment)
# word_index = tokenizer.word_index
# print(word_index)

train_sequences = tokenizer.texts_to_sequences(train_sentiment)
train_pad_sequences = pad_sequences(train_sequences, padding='post', maxlen=max_length, truncating='post')

test_sequences = tokenizer.texts_to_sequences(test_sentiment)
test_pad_sequences = pad_sequences(test_sequences, padding='post', maxlen=max_length, truncating='post')

# print(train_pad_sequences[0])
# print(test_pad_sequences[0])

model_lstm = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_lstm.summary()


# compile and fit the model
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_lstm.fit(train_pad_sequences, train_labels, validation_data=(test_pad_sequences, test_labels), epochs=20,
                         verbose=2)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.show()


