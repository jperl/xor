# OpenAI [XOR Warmup](https://blog.openai.com/requests-for-research-2/)

⭐ Train an LSTM to solve the XOR problem: that is, given a sequence of bits, determine its parity. The LSTM should consume the sequence, one bit at a time, and then output the correct answer at the sequence’s end.

## Questions

Test the two approaches below:

- Generate a dataset of random 100,000 binary strings of length 50. Train the LSTM; what performance do you get?

It converges after 16,000 examples.

![](https://user-images.githubusercontent.com/1136652/43974327-39756404-9ca8-11e8-9fc7-a824d09f50d9.png)

- Generate a dataset of random 100,000 binary strings, where the length of each string is independently and randomly chosen between 1 and 50. Train the LSTM. Does it succeed? What explains the difference?

Yes. It converges after 4,000 examples. It converges faster is because it is easier to learn shorter sequences.

![](https://user-images.githubusercontent.com/1136652/43994346-54138e46-9d69-11e8-8c08-62acad6f1375.png)

## Getting Started

**Install dependencies**

    pip install -r requirements.txt

**Train an LSTM without varying lengths**

    python train.py --momentum 0.99

**Train an LSTM with varying lengths**

    python train.py --vary_lengths True --momentum 0.9

**Train a basic neural network written in Numpy to learn XOR on [colab](https://colab.research.google.com/github/hedgehoglabs/xor/blob/master/notebooks/NumpyXOR.ipynb)**
