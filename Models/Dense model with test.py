import numpy as np
import pandas as pd
import os
# Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten
from keras.utils import set_random_seed
# Preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# Train test split
from sklearn.model_selection import train_test_split

# Read glove embeddings and data
script_dir = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(script_dir, '..', 'data', 'glove.6B.50d.txt')
embeddings_index = dict()
with open(filepath, encoding="utf8") as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
file.close()

filepath = os.path.join(script_dir, '..', 'data', 'imdb_movie.zip')
data = pd.read_csv(filepath)

# Split into train and test data
labels = data['label'].array
reviews = data['text'].array
x_train, x_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, stratify=labels, random_state=1)

# Tokenize and pad the train data
t = Tokenizer()
t.fit_on_texts(x_train)
vocab_size = len(t.word_index) + 1
encoded_x = t.texts_to_sequences(x_train)

padded_x = pad_sequences(encoded_x, padding='post')
input_length = padded_x.shape[1]
print(vocab_size, input_length)

# Tokenize and pad the test data with the same tokenizer and pad the same length as train data
encoded_x_test = t.texts_to_sequences(x_test)
padded_x_test = pad_sequences(encoded_x_test, maxlen=input_length, padding='post')
print(padded_x.shape)
print(padded_x_test.shape)

# Create embedding matrix based on train data
# This becomes part of the model, cannot use glove embeddings for words outside train data
embedding_matrix = np.zeros((vocab_size, 50))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Define model
x = padded_x.astype(np.float32)
y = y_train.astype(np.float32)
set_random_seed(1)
model = Sequential()
e = Embedding(input_dim=vocab_size, 
              output_dim=50, 
              weights=[embedding_matrix], 
              input_length=input_length, 
              trainable=True)
model.add(e)
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
print(model.summary())

model.fit(x, y, epochs=20, batch_size=32, verbose=1)
model.save('models/model_v1.h5')
