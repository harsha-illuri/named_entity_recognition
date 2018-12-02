
from dataPreProcessing import get_tagged_sentences, addCharInformatioin, padding, createMatrices, create_word_index

from keras.models import load_model

# create all maps
label_map = {'PADDING': 0, 'B-EVIDENTIAL': 1, 'B-OCCURRENCE': 2, 'B-TREATMENT': 3, 'B-TEST': 4, 'I-TEST': 5, 'B-CLINICAL_DEPT': 6, 'I-PROBLEM': 7, 'I-CLINICAL_DEPT': 8, 'B-PROBLEM': 9, 'none': 10, 'I-EVIDENTIAL': 11, 'I-TREATMENT': 12, 'I-OCCURRENCE': 13}
inv_map = {0: 'PADDING', 1: 'EVIDENTIAL', 2: 'OCCURRENCE', 3: 'TREATMENT', 4: 'TEST', 5: 'TEST', 6: 'CLINICAL_DEPT', 7: 'PROBLEM', 8: 'CLINICAL_DEPT', 9: 'PROBLEM', 10: 'none', 11: 'EVIDENTIAL', 12: 'TREATMENT', 13: 'OCCURRENCE'}
char_map = {"PADDING":0, "UNKNOWN":1}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
    char_map[c] = len(char_map)
word_map = create_word_index('data')
word_map['UNKNOWN_TOKEN'] = len(word_map)
word_map['PADDING_TOKEN'] = len(word_map)

# get output of each file - one at a time


import os
from shutil import copyfile, rmtree
text_files = os.listdir('test/txt')
text_files = [f for f in text_files if '.txt' in f]
model = load_model('ner_events_100.h5')

def genfile():
    sentences = get_tagged_sentences('test/temp')
    dev_sentences = addCharInformatioin(sentences)
    dev_tokens, dev_char, dev_labels = padding(createMatrices(dev_sentences, word_map, label_map, char_map))
    pred_labels = model.predict([dev_tokens, dev_char])
    pred_labels = pred_labels.argmax(axis=-1)
    event_id = 0
    start_pos = 1
    for i in range(len(sentences)):
        l = min(100, len(sentences[i]))
        sent = sentences[i]
        pred = pred_labels[i][:l]
        sent_start_pos = start_pos
        i = 0
        while i < l:
            # print("sent_start_pos", sent_start_pos)
            # print(sent[i][0])
            # for each token
            if pred[i] != 10:
                attrib = {'id': 'E0', 'start': '1', 'end': '10', 'text': 'Admission', 'modality': 'FACTUAL',
                          'polarity': 'POS', 'type': 'OCCURRENCE'}
                attrib['start'] = str(sent_start_pos)
                attrib['text'] = sent[i][0]
                j = i + 1
                # print(i, j)
                # print(pred[i])
                # print(inv_map[pred[i]])
                while j < l and inv_map[pred[i]] == inv_map[pred[j]]:
                    attrib['text'] += " " + sent[j][0]
                    sent_start_pos += len(sent[j][0]) + 1
                    j += 1
                sent_start_pos += len(sent[i][0]) + 1
                attrib['end'] = str(sent_start_pos)

                attrib['id'] = "E" + str(event_id)
                event_id += 1
                attrib['type'] = inv_map[pred[i]]
                i = j - 1
                t = "<EVENT "
                for k, v in attrib.items():
                    t += k + "=" + "\"" + v + "\"" + " "
                t += "/>"
                out.write(t + "\n")

            else:
                sent_start_pos += len(sent[i][0]) + 1
            i += 1
            # print("end sent_start_pos", sent_start_pos)

        while (i < len(sent)):
            sent_start_pos += len(sent[i][0]) + 1
            i += 1
        start_pos = sent_start_pos
    out.write("""</TAGS>\n</ClinicalNarrativeTemporalAnnotation>""")
import gc
import multiprocessing
import time

for i in range(len(text_files)):
    try:
        rmtree('test/temp')
    except FileNotFoundError:
        pass
    os.mkdir('test/temp')
    copyfile('test/txt/'+text_files[i], 'test/temp/'+text_files[i])
    out = open('test/gen/'+text_files[i][:-4], 'w')
    out.write("""<?xml version="1.0" encoding="UTF-8" ?>\n
<ClinicalNarrativeTemporalAnnotation>\n
<TEXT></TEXT>\n<TAGS>\n""")
    genfile()
    gc.collect()