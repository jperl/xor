import numpy as np
from numpy.testing import assert_equal
from xor_dataset import get_random_bits_parity

np.random.seed(0)


def test_get_random_bits_parity():
  bit_sequences, parity = get_random_bits_parity(num_sequences=5, num_bits=5)

  assert_equal(
      bit_sequences,
      [
          # sum 3 -> odd -> parity 1
          [0, 1, 1, 0, 1],
          # sum 5 -> odd -> parity 1
          [1, 1, 1, 1, 1],
          # sum 2 -> even -> parity 0
          [1, 0, 0, 1, 0],
          # sum 1 -> odd -> parity 1
          [0, 0, 0, 0, 1],
          # sum 2 -> even -> 0
          [0, 1, 1, 0, 0]
      ])

  assert_equal(parity, [1, 1, 0, 1, 0])
