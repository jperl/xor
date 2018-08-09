import numpy as np
from numpy.testing import assert_equal
from xor_dataset import get_random_bits_parity

np.random.seed(0)


def test_get_random_bits_parity():
  bit_sequences, parity = get_random_bits_parity(num_sequences=5, num_bits=5)

  assert_equal(
      bit_sequences,
      [
          #even, odd, even, even, odd
          [0, 1, 1, 0, 1],
          #odd, even, odd, even, odd
          [1, 1, 1, 1, 1],
          # odd, odd, odd, even, even
          [1, 0, 0, 1, 0],
          # even, even, even, even, odd
          [0, 0, 0, 0, 1],
          # even, odd, even, even, even
          [0, 1, 1, 0, 0]
      ])

  assert_equal(parity, [
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0],
  ])
