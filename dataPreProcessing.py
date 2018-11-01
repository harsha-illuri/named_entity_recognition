import pandas as pd
import numpy as np
import os

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
                tokens.append([token, tokennum, linenum + 1])
        df = pd.DataFrame(tokens)
        df.columns = ['text', 'position', 'line_num']
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
                            data_per_file.append([event_text[i], pos, line, event_type, modality, polarity])
                            i += 1
            df_extent = pd.DataFrame(data_per_file)
            df_extent.columns = ['text', 'position', 'line_num', 'event_type', 'modality', 'polarity']
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



    def process_times_extent_file(self):
        pass

    def get_merged_time_events(self):
        # returns dataframe
        pass



def getCasing(word, caseLookup):
        casing = 'other'

        numDigits = 0
        for char in word:
            if char.isdigit():
                numDigits += 1

        digitFraction = numDigits / float(len(word))

        if word.isdigit():  # Is a digit
            casing = 'numeric'
        elif digitFraction > 0.5:
            casing = 'mainly_numeric'
        elif word.islower():  # All lower case
            casing = 'allLower'
        elif word.isupper():  # All upper case
            casing = 'allUpper'
        elif word[0].isupper():  # is a title, initial char upper, then all lower
            casing = 'initialUpper'
        elif numDigits > 0:
            casing = 'contains_digit'

        return caseLookup[casing]
