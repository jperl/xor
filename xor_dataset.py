import numpy as np
import os
import torch
import torch.utils.data as data

from utils import ensure_path, remove_path

DEFAULT_NUM_BITS = 5
DEFAULT_NUM_SEQUENCES = 100000


class XORDataset(data.Dataset):
  data_folder = './data'

  def __init__(self, train=True, test_size=0.2):
    self._test_size = test_size
    self.train = train

    # cache dataset so training is deterministic
    self.ensure_sequences()

    filename = 'train.pt' if self.train else 'test.pt'
    self.features, self.labels = torch.load(f'{self.data_folder}/{filename}')

  def ensure_sequences(self):
    if os.path.exists(self.data_folder):
      return

    ensure_path(self.data_folder)

    features, labels = get_random_bits_parity()

    test_start = int(len(features) * (1 - self._test_size))

    train_set = (features[:test_start], labels[:test_start])
    test_set = (features[test_start:], labels[test_start:])

    with open(f'{self.data_folder}/train.pt', 'wb') as file:
      torch.save(train_set, file)

    with open(f'{self.data_folder}/test.pt', 'wb') as file:
      torch.save(test_set, file)

  def __getitem__(self, index):
    return self.features[index, :], self.labels[index]

  def __len__(self):
    return len(self.features)


def get_random_bits_parity(num_sequences=DEFAULT_NUM_SEQUENCES, num_bits=DEFAULT_NUM_BITS):
  """Generate random bit sequences and their parity. (Our features and labels).
    Returns:
      bit_sequences: A numpy array of bit sequences with shape [num_sequences, num_bits].
      parity: A numpy array of even parity values corresponding to each bit
        with shape [num_sequences, num_bits].
    """
  bit_sequences = np.random.randint(2, size=(num_sequences, num_bits))

  # if total number of ones is odd, set even parity bit to 1, otherwise 0
  # https://en.wikipedia.org/wiki/Parity_bit


  bitsum = np.cumsum(bit_sequences, axis=1)
  # if bitsum is even: False, odd: True
  parity = bitsum % 2 != 0

  return bit_sequences.astype('float32'), parity.astype('float32')


if __name__ == '__main__':
  remove_path(XORDataset.data_folder)
  XORDataset(test_size=0.2)
