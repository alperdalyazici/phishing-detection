import json
import random
import numpy as np

class DataProcess:
    def __init__(self):
        self.legit_data = "data/data_legitimate_36400.json"
        self.phish_data = "data/data_phishing_37175.json"
        self.legit_url = []
        self.phish_url = []
        self.train_data = []
        self.test_data = []
        self.val_data = []
        self.all_data = []

    #This function is used for reading the legitimate URLs and convert them into a list of tuples with the label 0
    def read_legit_data(self):
        with open(self.legit_data, 'r', encoding = 'utf-8') as f:
            urls = json.load(f)
        for url in urls:
            self.legit_url.append((url, 0))
        
    #This function is used for reading the phishing URLs and convert them into a list of tuples with the label 1
    def read_phish_data(self):
        with open(self.phish_data, 'r', encoding = 'utf-8') as f:
            urls = json.load(f)
        for url in urls:
            self.phish_url.append((url, 1))
    
    #This function is used for combining the legitimate and phishing URLs and shuffle them
    def shuffle_data(self):
        self.all_data = self.legit_url + self.phish_url
        random.shuffle(self.all_data)

    #This function is used for splitting the data into training, testing, and validation sets
    def split_data(self, train_frac=0.8, test_frac=0.1, val_frac=0.1):
        num_samples = len(self.all_data)
        num_train = int(train_frac * num_samples)
        num_test = int(test_frac * num_samples)
        num_val = num_samples - num_train - num_test

        # Split the data into training, testing, and validation sets
        self.train_data = self.all_data[:num_train]
        self.val_data = self.all_data[num_train:num_train + num_val]
        self.test_data = self.all_data[num_train + num_test:]
     
        
    def get_features_labels(self, data):
        urls, labels = zip(*data)
        return np.array(urls), np.array(labels)


    def txt_to_json(self,input_file, output_file):
        sentences = []

        # Step 1: Read the TXT file and extract the data with 'utf-8' encoding
        with open(input_file, 'r', encoding='utf-8') as txt_file:
            for line in txt_file:
                sentence = line.strip()

                # Skip empty lines
                if sentence:
                    sentences.append(sentence)

 
        # Step 3: Write the data to a new JSON file
        with open(output_file, 'w') as json_file:
            json.dump(sentences, json_file, indent=4)
