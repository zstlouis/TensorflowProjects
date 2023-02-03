import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

tokenizer = Tokenizer()
# read data from poerty sample
data = open('poetry_sample').read()

# store text as this will be used to train the model
# will be all possible words available
corpus = data.lower().split('\n')
print(corpus)

# tokenizer the text
tokenizer.fit_on_texts(corpus)

word_idx = tokenizer.word_index

# store the total number of words
total_words = len(word_idx) + 1
print(word_idx)
print(total_words)

input_sequence = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        input_sequence.append(n_gram_seq)

# pad sequences
max_sequence_len = max([len(x) for x in input_sequence])
input_sequence = np.array(pad_sequences(input_sequence, maxlen=max_sequence_len, padding='pre'))

# create predictors and labels
xs, labels = input_sequence[:,:-1], input_sequence[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)


model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len-1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(xs, ys, epochs=75, verbose=1)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()


seed_text = "I've got a bad feeling about this"
next_words = 50

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word

with open('generated_poetry', 'w') as f:
    f.write(seed_text)
f.close()
print(seed_text)
