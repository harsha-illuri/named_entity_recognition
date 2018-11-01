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
            f = open(file_name)
            sentences.append([line for line in f])
        return sentences

    def process_text_files(self):
        # returns dataframe
        tagged_text = []
        for file_name in self.files_txt:
            with open(file_name, 'r') as f:
                data = f.read().splitlines()
            tokens = []
            for linenum, line in enumerate(data):
                for tokennum, token in enumerate(line.split()):
                    tokens.append([token, tokennum, linenum + 1])
            tagged_text += tokens
        return pd.DataFrame(tagged_text)

    def process_tlink_files(self):
        # returns dataframe
        for file_name in self.files_extent:
            extracted_text = []
            with open(file_name, 'r') as data:
                data_per_file = []
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
                        sec_time_rel = parts[4][14:-1]
                        event_text = event_text.strip()
                        event_text = event_text.split(' ')
                        i = 0
                        for line in range(event_start_line, event_end_line + 1):
                            for pos in range(event_start_token, event_end_token + 1):
                                data_per_file.append([event_text[i], pos, line, event_type, modality, polarity, sec_time_rel])
                                i += 1
                    else:
                        pass
                extracted_text += data_per_file
            return pd.DataFrame(extracted_text)

    def process_events_extent_files(self):
        # returns dataframe
        for file_name in self.files_tlink:
            extracted_text = []
            with open(file_name, 'r') as data:
                data_per_file = []
                for line in data:
                    pass
                    # @others write code for processing the tlink files
                    # if "EVENT" in line:
                    #     parts = line.split("||")
                    #     # get event name and time
                    #     event_text = parts[0][parts[0].find("EVENT=") + 7:parts[0].rfind('\"')]
                    #     event = parts[0][parts[0].rfind('\"') + 2:]
                    #     event = event.split()
                    #     event_start_line = int(event[0].split(':')[0])
                    #     event_start_token = int(event[0].split(':')[1])
                    #     event_end_line = int(event[1].split(':')[0])
                    #     event_end_token = int(event[1].split(':')[1])
                    #     event_type = parts[1][6:-1]
                    #     modality = parts[2][10:-1]
                    #     polarity = parts[3][10:-1]
                    #     sec_time_rel = parts[4][14:-1]
                    #     event_text = event_text.strip()
                    #     event_text = event_text.split(' ')
                    #     i = 0
                    #     for line in range(event_start_line, event_end_line + 1):
                    #         for pos in range(event_start_token, event_end_token + 1):
                    #             data_per_file.append([event_text[i], pos, line, event_type, modality, polarity, sec_time_rel])
                    #             i += 1
                    # else:
                    #     pass
                extracted_text += data_per_file
            return pd.DataFrame(extracted_text)

    def process_times_extent_files(self):
        # returns dataframe
        for file_name in self.files_extent:
            extracted_text = []
            with open(file_name, 'r') as data:
                data_per_file = []
                for line in data:
                    if "TIMEX3" in line:
                        pass
                        #@Amulya change below code for processing the tim eevents in extent


                        # parts = line.split("||")
                        # # get event name and time
                        # event_text = parts[0][parts[0].find("EVENT=") + 7:parts[0].rfind('\"')]
                        # event = parts[0][parts[0].rfind('\"') + 2:]
                        # event = event.split()
                        # event_start_line = int(event[0].split(':')[0])
                        # event_start_token = int(event[0].split(':')[1])
                        # event_end_line = int(event[1].split(':')[0])
                        # event_end_token = int(event[1].split(':')[1])
                        # event_type = parts[1][6:-1]
                        # modality = parts[2][10:-1]
                        # polarity = parts[3][10:-1]
                        # sec_time_rel = parts[4][14:-1]
                        # event_text = event_text.strip()
                        # event_text = event_text.split(' ')
                        # i = 0
                        # for line in range(event_start_line, event_end_line + 1):
                        #     for pos in range(event_start_token, event_end_token + 1):
                        #         data_per_file.append([event_text[i], pos, line, event_type, modality, polarity, sec_time_rel])
                        #         i += 1
                    else:
                        pass
                extracted_text += data_per_file
            return pd.DataFrame(extracted_text)
