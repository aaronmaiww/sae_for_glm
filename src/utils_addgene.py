import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


def split_test_data(test_data):
    """Split test data into input and target variables."""
    y_test = test_data['nations']
    x_test = test_data[['sequence']]
    return x_test, y_test

def replace_infrequent_labels(labels, threshold:int = 10):
    """Identify and replace infrequent labels."""
    label_counts = labels.value_counts()
    infrequent_labels = label_counts[label_counts < threshold].index
    return labels.replace(infrequent_labels, 'infrequent')

def map_labels_to_integers(labels):
    """Map labels to integers."""
    unique_labels = labels.unique()
    return {label: int(i) for i, label in enumerate(unique_labels)}

def without_US(data):
    """Filter out rows where the nation is 'UNITED STATES'."""
    data_wo_US = data[data['nations'] != 'UNITED STATES']
    data_wo_US.reset_index(drop=True, inplace=True)

    data_w_US = data[data['nations'] == 'UNITED STATES']
    data_w_US.reset_index(drop=True, inplace=True)
    return data_wo_US, data_w_US

def US_vs_them(labels):
    """Categorize labels into 'UNITED STATES' and 'NON US'."""
    return labels.apply(lambda x: x if x == 'UNITED STATES' else 'NON US')

def pad_sequence(seq, length, pad_char='N'):
    """Pad sequences to the specified length with the given character."""
    return seq.ljust(length, pad_char)[:length]


def preprocess_data(train_data_path:str, test_data_path:str, min_length:int = 0):
    """Process the data."""
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    x_train, y_train = train_data[['sequence']], train_data['nations']
    x_test, y_test = split_test_data(test_data)

    # Combine labels from train and test datasets
    processed_labels = pd.concat([y_train, y_test], axis=0, ignore_index=True)
    label_to_int = map_labels_to_integers(processed_labels)

    # map labels to integers
    y_train = y_train.map(label_to_int)
    y_test = y_test.map(label_to_int)

    print(f'y_test shape: {y_test.shape}')

    # reset indices before concat
    x_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    df_train = pd.concat([x_train, y_train], axis=1)
    df_val = pd.concat([x_test, y_test], axis=1)

    # Filter out sequences shorter than min_length and clean them
    min_length = 0
    df_train = df_train[df_train['sequence'].str.len() > min_length]
    df_val = df_val[df_val['sequence'].str.len() > min_length]

    print(f'test_data shape: {test_data.shape}')


    # Ensure indices are reset correctly
    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)

    return df_train, df_val


class GenomicDataset(Dataset):
    def __init__(self,
                 ds: pd.DataFrame,
                 tokenizer_nt,
                 seq_length: int = 8000):


        self.sequences = ds['sequence']
        self.labels = ds['nations']
        self.seq_len = seq_length
        self.tokenizer = tokenizer_nt



    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences.iloc[idx]
        label = self.labels.iloc[idx]

        # Tokenize the sequence
        inputs = self.tokenizer(sequence, max_length=512, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'].squeeze()  # Remove batch dimension
        attention_mask = inputs['attention_mask'].squeeze()  # Remove batch dimension

        # to torch tensors
        label = torch.tensor(label, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }

