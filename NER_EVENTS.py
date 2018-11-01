import pandas as pd
import numpy as np
import os

import gensim
from gensim.models import Word2Vec

# get data

def clean_ner(row):
    try:
        return row.lower()
    except AttributeError:
        return "None"

def ceate_map_categories(categories, vals):
    d = {}
    categories.sort()
    for i,v in enumerate(categories):
        d[v] = i
    return [d[v] for v in vals]

from dataPreProcessing import PreProcessFiles
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Reshape, Bidirectional, concatenate, Flatten
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
test = PreProcessFiles('data')
data = test.get_merged_events()
data = data[:1000].copy()
data['event_type'] = data['event_type'].apply(clean_ner)
sentences = test.get_sentences()

max_words = 100
# tokenizer
tokenizer = Tokenizer(num_words= max_words, lower=False)
tokenizer.fit_on_texts(sentences)
X_words = tokenizer.texts_to_matrix(data['text_x'].values, mode='count')

num_categories = len(set(data['event_type'].values))
categories = list(set(data['event_type'].values))

Y = ceate_map_categories(categories, data['event_type'].values)
Y = to_categorical(Y, num_categories)

print(len(Y))
print(len(Y[0]))

X_words = X_words.reshape(1, len(X_words), 1)
Y = X_words.reshape(1, len(Y), 1)
model = Sequential()
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(len(X_words), 1)))
model.add(TimeDistributed(Dense(num_categories, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

#
# X_words = X_words.reshape(len(X_words),max_words, 1)
# model = Sequential()
# model.add(LSTM(20, input_shape=(max_words,1)))
# model.add(Dense(num_categories, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model = Sequential()
# model.add(Dense(512, input_shape=(1000,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_categories))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.metrics_names)
#
batch_size = 16
epochs = 10

history = model.fit(X_words, Y, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2)



# sentences = [gensim.utils.simple_preprocess(line) for line in sentences]
# model = Word2Vec(sentences, size=150, window=10, min_count=2, workers=10)
# model.train(sentences, total_examples=len(sentences), epochs=10)
#
#
# words = list(model.wv.vocab)
# print(words)
#
# w = 'pregnancy'
# print(model.wv.most_similar(positive = w, topn=10))

def load_data(d, word_tokenizer=None, ner_tokenizer=None, char_tokenizer=None):
    sents = data['text'].values
    ner = data['event_type'].values

    chars = pprc.split_words(sents, padding=True, pad_len=max_word)
    tokenized_words, word_tokenizer = pprc.tokenize(sents, t=word_tokenizer)
    one_hot_ner, ner_tokenizer = pprc.one_hot_encode(ner, t=ner_tokenizer)
    tokenized_chars, char_tokenizer = pprc.tokenize_chars(chars)
    return (tokenized_words, word_tokenizer), (one_hot_ner, ner_tokenizer), (tokenized_chars, char_tokenizer)