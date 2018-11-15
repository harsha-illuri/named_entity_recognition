import pandas as pd
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def fetch_minibatch(dataset,batch_indices):
    start = 0
    for i in batch_indices:
        tokens = []
        char = []
        labels = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t,ch,l = dt
            l = np.expand_dims(l,-1)
            tokens.append(t)
            char.append(ch)
            labels.append(l)
        yield np.asarray(labels),np.asarray(tokens),np.asarray(char)

def get_accuracy(pred_labels, orig_labels):
    # calculate accuray of dev set
    label_pred = []
    for sentence in pred_labels:
        for word in sentence:
            label_pred.append(word)

    label_correct = []
    for sentence in orig_labels:
        sentence = [s[0] for s in sentence]
        for word in sentence:
            label_correct.append(word)
    # print(len(pred_labels), len(orig_labels))
    # print(len(label_pred), len(label_correct))
    corerct = 0
    temp = 0
    for i in range(len(label_correct)):
        if label_pred[i] == label_correct[i]:
            corerct +=1
            if label_pred[i] == 0 and label_correct[i] == 0:
                temp +=1
    return (corerct/len(label_pred), (corerct-temp)/(len(label_pred)-temp))

def defineBatches(data):
    # since the sentences are of multiple sizes, instead of padding
    # train on batches of data with same length

    #split the dataset in to multiple mini batches
    l = []
    for i in data:
        l.append(len(i[0]))
    l = set(l)
    batches = []
    batch_len = []
    z = 0
    for i in l:
        for batch in data:
            if len(batch[0]) == i:
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches, batch_len

def padding(Sentences):
    # add padding to char vectors
    word_len = 20
    sentence_len = 25
    # print(Sentences[0])
    for i,sentence in enumerate(Sentences):
        Sentences[i][1] = pad_sequences(Sentences[i][1],word_len,padding='post')
    tokens = []
    labels = []
    char = []
    for sent in Sentences:
        tokens.append(sent[0])
        char.append(sent[1])
        l = sent[2]
        l = np.expand_dims(l, -1)
        labels.append(l)
    # return tokens, char, labels
    tokens = pad_sequences(tokens,sentence_len,padding='post')
    char = pad_sequences(char,sentence_len,padding='post')
    labels = pad_sequences(labels,sentence_len,padding='post')
    return np.asarray(tokens), np.asarray(char), np.asarray(labels)


def createMatrices(sentences, word_map, label_map, char2Idx):
    unknownIdx = word_map['UNKNOWN_TOKEN']
    paddingIdx = word_map['PADDING_TOKEN']
    dataset = []
    wordCount = 0
    unknownWordCount = 0

    for sentence in sentences:
        wordIndices = []
        charIndices = []
        labelIndices = []

        for word, char, label in sentence:
            wordCount += 1
            if word in word_map:
                wordIdx = word_map[word]
            elif word.lower() in word_map:
                wordIdx = word_map[word.lower()]
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1
            charIdx = []
            for x in char:
                if x in char2Idx:
                    charIdx.append(char2Idx[x])
                else:
                    charIdx.append(char2Idx['UNKNOWN'])
            # Get the label and map to int
            wordIndices.append(wordIdx)
            charIndices.append(charIdx)
            labelIndices.append(label_map[label])

        dataset.append([wordIndices, charIndices, labelIndices])
    return dataset

def create_word_index(file_path, vocab_size=50, num_words = 1000, max_sentence_length = 50):
    class_obj = PreProcessFiles(file_path)
    sentences = class_obj.get_sentences()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    return tokenizer.word_index

def get_labels_words(sentences):
    labelSet = set()
    words = {}
    for sentence in sentences:
        for token, char, label in sentence:
            # print(sentence)
            labelSet.add(label)
            words[token.lower()] = True
    return labelSet, words

def addCharInformatioin(Sentences):
    for i,sentence in enumerate(Sentences):
        for j,data in enumerate(sentence):
            chars = [c for c in data[0]]
            Sentences[i][j] = [data[0],chars,data[1]]
    return Sentences

def get_tagged_sentences(file_path):
    class_obj = PreProcessFiles(file_path)
    data = class_obj.get_merged_events()
    data.fillna('none', inplace=True)
    data['text_x'] = data['text_x'].str.lower()
    grouped = data.groupby(['file_name_x', 'line_num'])
    sentences = []
    for name, group in grouped:
        sentence = []
        for row, d in group.iterrows():
            sentence.append([d['text_x'], d['event_type']])
        sentences.append(sentence)
    return sentences

class PreProcessFiles():
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)
        self.files_txt = [self.path+'/'+_ for _ in self.files if ".txt" in _]
        self.files_extent = [self.path+'/'+_ for _ in self.files if ".extent" in _]
        self.files_tlink = [self.path+'/'+_ for _ in self.files if ".tlink" in _]

    def get_sentences(self):
        # returns array
        sentences = []
        for file_name in self.files_txt:
            f = open(file_name)
            for line in f:
                sentences.append(line)
        return sentences

    def get_text(self):
        # returns array
        sentences = []
        for file_name in self.files_txt:
            fle = []
            f = open(file_name)
            for line in f:
                fle.append(line)
            sentences.append(''.join(fle))
        return sentences

    def process_text_file(self, file_name):
        with open(file_name, 'r') as f:
            data = f.read().splitlines()
        tokens = []
        for linenum, line in enumerate(data):
            for tokennum, token in enumerate(line.split()):
                tokens.append([token, tokennum, linenum + 1, file_name])
        df = pd.DataFrame(tokens)
        df.columns = ['text', 'position', 'line_num', 'file_name']
        return df

    def process_events_extent_file(self, file_name):
        data_per_file = []
        with open(file_name, 'r') as data:
            for line in data:
                if "EVENT" in line:
                    parts = line.split("||")
                    # get event name and time
                    event_text = parts[0][parts[0].find("EVENT=") + 7:parts[0].rfind('\"')]
                    event = parts[0][parts[0].rfind('\"') + 2:]
                    event = event.split()
                    event_start_line = int(event[0].split(':')[0])
                    event_start_token = int(event[0].split(':')[1])
                    event_end_line = int(event[1].split(':')[0])
                    event_end_token = int(event[1].split(':')[1])
                    event_type = parts[1][6:-1]
                    modality = parts[2][10:-1]
                    polarity = parts[3][10:-1]
                    # sec_time_rel = parts[4][14:-1]
                    event_text = event_text.strip()
                    event_text = event_text.split(' ')
                    i = 0
                    for line in range(event_start_line, event_end_line + 1):
                        for pos in range(event_start_token, event_end_token + 1):
                            # print([event_text[i], pos, line, event_type, modality, polarity, sec_time_rel])
                            data_per_file.append([event_text[i], pos, line, event_type, modality, polarity, file_name])
                            i += 1
            df_extent = pd.DataFrame(data_per_file)
            df_extent.columns = ['text', 'position', 'line_num', 'event_type', 'modality', 'polarity', 'file_name']
            return pd.DataFrame(df_extent)

    def get_merged_events(self):
        dfs = []
        for i in range(len(self.files_extent)):
            extent = self.files_extent[i]
            txt = extent[:extent.rfind(".extent")]+".txt"
            df = pd.merge(self.process_text_file(txt), self.process_events_extent_file(extent), on=['position', 'line_num'], how = 'left')
            dfs.append(df)
        res = pd.concat(dfs, ignore_index=True)
        return res

