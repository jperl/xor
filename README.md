# OpenAI [XOR Warmup](https://blog.openai.com/requests-for-research-2/)

⭐ Train an LSTM to solve the XOR problem: that is, given a sequence of bits, determine its parity. The LSTM should consume the sequence, one bit at a time, and then output the correct answer at the sequence’s end.

## Questions

Test the two approaches below:

- Generate a dataset of random 100,000 binary strings of length 50. Train the LSTM; what performance do you get?

It converges after 16,000 examples.

![](https://user-images.githubusercontent.com/1136652/43974327-39756404-9ca8-11e8-9fc7-a824d09f50d9.png)

- Generate a dataset of random 100,000 binary strings, where the length of each string is independently and randomly chosen between 1 and 50. Train the LSTM. Does it succeed? What explains the difference?

## Getting Started

Install Dependencies

    pip install -r requirements.txt

Train Approach One

    python train.py
