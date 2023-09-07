## Introduction
This project creates a sentiment analysis model on the IMDB movie review dataset. There are two columns: the text of the review, and the sentiment of the review 1 (positive) or 0.(negative).

Each unique word in the entire dataset is mapped to an index of words. The words in each review are represented with the word's number in the index (Bag of words model). Each index number is encoded with a word embeddings which is a real-valued vector in a high dimensional space that represents the meaning of the word. A review with N words represented by M-dimension word vectors will for a vector of NxM values.

Future versions of this project can utilise a tf-idf model instead of a BoW model and limit the number of encoded words to reduce complexity and improve accuracy of the sentiment analysis.

I perform this project on two scales, the full scale and a partial scale. 

## Workflow
The process involves:
- Assembling data (Importing, splitting)
- Tokenisation and representing each unique word as an integer
- Padding data to desired length
- Building the model with an embedding layer
- Training the model

In the partial scale only the top 5000 unique words are represented as integers with a maximum length of 500 words in a review, and these words are encoded with a vector of 32 dimensions. Any words that are not represented in the index or encoded show up as 0s. The training time of this scale is SIGNIFICANTLY lower that a full scale model because it is a far much less complex model.

On the full scale. The review data is imported directly from the raw textfile dataset and the GloVe word embeddings 50 dimensions file. The data is split into the train and test data with a 80-20 split. In this example I only encode word embeddings for words in the training data by entering the embedding matrix created form the train data into the emebdding layer of the model. The maximum length of an input vector is the length of the longest review in the train data (2332 words), both the train and test data are padded to this length. 

## Full IMDB dataset with full word embeddings and vocabulary
In the full scale exercise, models with RNN or CNN layers have poor accuracy due to the low neuron numbers. Low neuron numbers are necessary because training a highly complex model of this size requires GPU-acceleration which I do not use.
- Baseline model
- Dense layer model (Simple feed forward network)
- LSTM model (poor accuracy)
- RNN and CNN model (poor accuracy)

## Imported data with partial word embeddings and vocabulary
- LSTM model