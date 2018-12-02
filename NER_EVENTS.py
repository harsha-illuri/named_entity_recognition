# imports
from dataPreProcessing import get_tagged_sentences,    get_labels_words,  addCharInformatioin, create_word_index,createMatrices,    padding, defineBatches, fetch_minibatch
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

label_map = {'PADDING': 0, 'B-EVIDENTIAL': 1, 'B-OCCURRENCE': 2, 'B-TREATMENT': 3, 'B-TEST': 4, 'I-TEST': 5, 'B-CLINICAL_DEPT': 6, 'I-PROBLEM': 7, 'I-CLINICAL_DEPT': 8, 'B-PROBLEM': 9, 'none': 10, 'I-EVIDENTIAL': 11, 'I-TREATMENT': 12, 'I-OCCURRENCE': 13}
inv_map = {0: 'PADDING', 1: 'EVIDENTIAL', 2: 'OCCURRENCE', 3: 'TREATMENT', 4: 'TEST', 5: 'TEST', 6: 'CLINICAL_DEPT', 7: 'PROBLEM', 8: 'CLINICAL_DEPT', 9: 'PROBLEM', 10: 'none', 11: 'EVIDENTIAL', 12: 'TREATMENT', 13: 'OCCURRENCE'}

#pre process the data
train_tokens, train_char, train_labels = padding(createMatrices(train_sentences,word_map,label_map, char_map))



# word embedding
words_input = Input(shape=(None,), dtype='int32', name='words_input')
words = Embedding(input_dim=len(word_map), output_dim=100)(words_input)

# charecter embedding
character_input = Input(shape=(None, 20,), name='char_input')
embed_char_out = TimeDistributed(
    Embedding(len(char_map), 500, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
    name='char_embedding')(character_input)

#char CNN
# dropout = Dropout(0.3)(embed_char_out)
# conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same', activation='tanh', strides=1))(dropout)
# maxpool_out = TimeDistributed(MaxPooling1D(20))(conv1d_out)
# char = TimeDistributed(Flatten())(maxpool_out)
# char = Dropout(0.3)(char)
# char = TimeDistributed(Embedding(input_dim=len(char_map), output_dim = 100))(embed_char_out)

# char BiLSTM
char = TimeDistributed(Bidirectional(LSTM(50, return_sequences=True),merge_mode='concat'))(embed_char_out)
char = TimeDistributed(Flatten())(char)

# main model
main_model_input = concatenate([words, char])
BiLSTM = Bidirectional(LSTM(50, return_sequences=True))(main_model_input)
# BiLSTM = Bidirectional(LSTM(20, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(main_model_input)
output = TimeDistributed(Dense(len(label_map), activation='softmax'))(BiLSTM)
# crf = CRF(len(label_map))
# output = crf(softmax)
model = Model(inputs=[words_input, character_input], outputs=[output])
# model.compile(loss=crf.loss_function,optimizer='nadam')
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
model.summary()

model.fit([train_tokens, train_char], train_labels, batch_size=128,epochs=25)
model.save('ner_events_100.h5')




