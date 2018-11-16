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

# create a dict to map the classes to numbers
# label_map = {}
# label_map['PADDING'] = 0
# for label in labels:
#     label_map[label] = len(label_map)
# print(label_map)
# create a inverse map of words for results
# inv_map = {}
# for k, v in label_map.items():
#     if '-' in k:
#         k = k[2:]
#     inv_map[v] = k
# print (inv_map)

#pre process the data
train_tokens, train_char, train_labels = padding(createMatrices(train_sentences,word_map,label_map, char_map))

# dev_sentences = get_tagged_sentences('dev')
# dev_sentences = addCharInformatioin(dev_sentences)
# dev_tokens, dev_char, dev_labels = padding(createMatrices(dev_sentences,word_map,  label_map, char_map))

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
output = Bidirectional(LSTM(50, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
output = TimeDistributed(Dense(len(label_map), activation='softmax'))(output)
# crf = CRF(len(label_map), name="output")
# output = crf(output)



model = Model(inputs=[words_input, character_input], outputs=[output])
# model.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
model.summary()


from keras.models import load_model
model.fit([train_tokens, train_char], train_labels, epochs=25)
model.save('ner_events_20.h5')


# def get_accuracy(pred_labels, orig_labels):
#     # calculate accuray of dev set
#     label_pred = []
#     for sentence in pred_labels:
#         for word in sentence:
#             label_pred.append(word)
#
#     label_correct = []
#     for sentence in orig_labels:
#         sentence = [s[0] for s in sentence]
#         for word in sentence:
#             label_correct.append(word)
#     corerct = 0
#     temp = 0
#     for i in range(len(label_correct)):
#         if inv_map[label_pred[i]] == inv_map[label_correct[i]]:
#             corerct +=1
#             if label_pred[i] == 0 and label_correct[i] == 0:
#                 temp +=1
#     return (corerct/len(label_pred), (corerct-temp)/(len(label_pred)-temp))
#
# #model = load_model('ner_events_20.h5')
# pred_labels = model.predict([dev_tokens, dev_char])
# pred_labels = pred_labels.argmax(axis=-1)
# print(get_accuracy(pred_labels, dev_labels))







