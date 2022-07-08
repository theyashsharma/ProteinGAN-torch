import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from common.bio.amino_acid import fasta_to_numpy
import json

SEQ_LENGTH = "seq_length"

def get_properties(path):
    with open(path) as json_data_file:
        properties = json.load(json_data_file)
    return properties

class Sequences(Dataset):
    def __init__(self, fasta_path, seq_length):
        self.data = nn.functional.one_hot(torch.tensor(fasta_to_numpy(fasta_path, seq_length)), -1)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]

def get_dataLoader(root_dir, batch_size):
    properties = get_properties(root_dir + 'properties.json')
    dataset = Sequences(root_dir + 'train_sequences.fasta', properties[SEQ_LENGTH])
    return DataLoader(dataset, batch_size=batch_size)

if __name__ == '__main__':
    # Testing
    dataset = get_dataLoader('../data/', 64)
