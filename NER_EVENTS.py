# imports
from dataPreProcessing import get_tagged_sentences,    get_labels_words,  get_accuracy,   addCharInformatioin, create_word_index,createMatrices,    padding, defineBatches, fetch_minibatch
from keras_contrib.layers import CRF
import numpy as np
from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
from keras.utils import Progbar
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform
import os
import matplotlib.pyplot as plt








# Hard coded charecter lookup
char_map = {"PADDING":0, "UNKNOWN":1}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
    char_map[c] = len(char_map)


# create a word to index map
word_map = create_word_index('data')
word_map['UNKNOWN_TOKEN'] = len(word_map)
word_map['PADDING_TOKEN'] = len(word_map)

# get the training set ready
train_sentences = get_tagged_sentences('data')
train_sentences = addCharInformatioin(train_sentences)
labels, words = get_labels_words(train_sentences)


# create a dict to map the classes to numbers
label_map = {}
label_map['PADDING'] = 0
for label in labels:
    label_map[label] = len(label_map)

#pre process the data
train_tokens, train_char, train_labels = padding(createMatrices(train_sentences,word_map,label_map, char_map))

dev_sentences = get_tagged_sentences('dev')
dev_sentences = addCharInformatioin(dev_sentences)
dev_tokens, dev_char, dev_labels = padding(createMatrices(dev_sentences,word_map,  label_map, char_map))

# load dev set
# dev_sentences = get_tagged_sentences('dev')
# dev_sentences = addCharInformatioin(dev_sentences)
# train_tokens, train_char, train_labels = padding(createMatrices(dev_sentences,word_map,  label_map, char_map))


# word embedding
words_input = Input(shape=(None,), dtype='int32', name='words_input')
words = Embedding(input_dim=len(word_map), output_dim=100)(words_input)

# charecter embedding
character_input = Input(shape=(None, 20,), name='char_input')
embed_char_out = TimeDistributed(
    Embedding(len(char_map), 120, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
    name='char_embedding')(character_input)
dropout = Dropout(0.3)(embed_char_out)
conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same', activation='tanh', strides=1))(dropout)
maxpool_out = TimeDistributed(MaxPooling1D(20))(conv1d_out)
char = TimeDistributed(Flatten())(maxpool_out)
char = Dropout(0.3)(char)

# main model
output = concatenate([words, char])
output = Bidirectional(LSTM(10, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
output = TimeDistributed(Dense(len(label_map), activation='softmax'))(output)

model = Model(inputs=[words_input, character_input], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
model.summary()

print(train_tokens[0])
print(train_char[0])
print(train_labels[0])

from keras.models import load_model
model.fit([train_tokens, train_char], train_labels, batch_size=32,epochs=30)
#model = load_model('ner_events_20.h5')
pred_labels = model.predict([dev_tokens, dev_char])
pred_labels = pred_labels.argmax(axis=-1)
print(get_accuracy(pred_labels, dev_labels))
model.save('ner_events_30.h5')






