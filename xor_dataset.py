import numpy as np
import os
import torch
import torch.utils.data as data

DEFAULT_NUM_BITS = 50
DEFAULT_NUM_SEQUENCES = 100000

np.random.seed(0)


class XORDataset(data.Dataset):
  data_folder = './data'

  def __init__(self):
    self.features, self.labels = get_random_bits_parity()

    # expand the dimensions for the lstm
    # [batch, bits] -> [batch, bits, 1]
    self.features = np.expand_dims(self.features, -1)

    # [batch, parity] -> [batch, parity, 1]
    self.labels = np.expand_dims(self.labels, -1)

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
