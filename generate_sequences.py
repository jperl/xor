import numpy as np
import os
import torch
import torch.utils.data as data

from utils import ensure_path, remove_path

DEFAULT_SEQUENCE_LENGTH = 50
NUM_SEQUENCES = 100000

class XORDataset(data.Dataset):
    data_folder = './data'

    def __init__(self, train=True, test_size=0.2):
        self._test_size = test_size
        self.train = train

        self.ensure_sequences()

        filename = 'train.pt' if self.train else 'test.pt'
        self.features, self.labels = torch.load(f'{self.data_folder}/{filename}')

    def ensure_sequences(self):
        if os.path.exists(self.data_folder):
            return

        ensure_path(self.data_folder)

        features, labels = generate_random_sequences()

        test_start = int(len(features) * (1 - self._test_size))

        train_set = (features[:test_start], labels[:test_start])
        test_set = (features[test_start:], labels[test_start:])

        with open(f'{self.data_folder}/train.pt', 'wb') as file:
            torch.save(train_set, file)

        with open(f'{self.data_folder}/test.pt', 'wb') as file:
            torch.save(test_set, file)

    def __getitem__(self, index):
        return self.features[:, index], self.labels[index]

    def __len__(self):
        return len(self.features)

# Data dimensions: [sequence_length, num_sequences, num_features]
def generate_random_sequences(sequence_length=DEFAULT_SEQUENCE_LENGTH, num_sequences=NUM_SEQUENCES):
    # generates num_sequences random bit sequences of length
    # extra dimension is num_features for pytorch, in this case 1
    sequences = np.random.randint(2, size=(sequence_length, num_sequences, 1))

    # if total number of ones is odd, odd parity bit set to 1, otherwise 0
    parity = (sequences.sum(axis=0) % 2 != 0).astype(int)

    return sequences, parity

if __name__ == '__main__':
    XORDataset(test_size=0.2)