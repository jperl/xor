import numpy as np
import torch
from torch.nn.utils import rnn as rnn_utils
from torch.utils.data import DataLoader
from typing import NamedTuple
from xor_dataset import XORDataset
from utils import get_arguments


class ModelParams(NamedTuple):
  # data
  max_bits: int = 50
  vary_lengths: bool = False

  # train loop
  batch_size: int = 8
  device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  epochs: int = 2
  momentum: float = 0

  # lstm
  hidden_size: int = 2
  lr: float = 1
  num_layers: int = 1


# make it deterministic
torch.manual_seed(0)


class LSTM(torch.nn.Module):
  def __init__(self, params: ModelParams):
    super().__init__()

    self._params = params

    self.lstm = torch.nn.LSTM(
        batch_first=True,
        input_size=1,
        hidden_size=params.hidden_size,
        num_layers=params.num_layers)

    self.hidden_to_logits = torch.nn.Linear(params.hidden_size, 1)
    self.activation = torch.nn.Sigmoid()

  def forward(self, inputs, lengths):
    # pack the inputs
    packed_inputs = rnn_utils.pack_padded_sequence(
        inputs, lengths, batch_first=True).to(params.device)

    lstm_out, _ = self.lstm(packed_inputs)

    unpacked, _ = rnn_utils.pad_packed_sequence(lstm_out, batch_first=True)

    logits = self.hidden_to_logits(unpacked)
    predictions = self.activation(logits)

    return logits, predictions


def train(params: ModelParams):
  model = LSTM(params).to(params.device)

  optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum)
  loss_fn = torch.nn.BCEWithLogitsLoss()

  train_loader = DataLoader(XORDataset(num_bits=params.max_bits), batch_size=params.batch_size)

  step = 0

  for epoch in range(1, params.epochs + 1):
    for inputs, targets in train_loader:
      lengths = adjust_lengths(params.vary_lengths, inputs, targets)
      targets = targets.to(params.device)
      optimizer.zero_grad()

      logits, predictions = model(inputs, lengths)

      # BCEWithLogitsLoss will do the activation
      loss = loss_fn(logits, targets)

      loss.backward()
      optimizer.step()
      step += 1

      accuracy = ((predictions > 0.5) == (targets > 0.5)).type(torch.FloatTensor).mean()

      if step % 250 == 0:
        print(f'epoch {epoch}, step {step}, loss {loss.item():.{4}f}, accuracy {accuracy:.{3}f}')

      if step % 1000 == 0:
        test_accuracy = evaluate(params, model)
        print(f'test accuracy {test_accuracy:.{3}f}')
        if test_accuracy == 1.0:
          # stop early
          break


def adjust_lengths(vary_lengths, inputs, targets):
  batch_size = inputs.size()[0]
  max_bits = inputs.size()[1]

  if not vary_lengths:
    lengths = torch.ones(batch_size, dtype=torch.int) * max_bits
    return lengths

  # choose random lengths
  lengths = np.random.randint(1, max_bits, size=batch_size, dtype=int)

  # keep one the max size so we don't need to resize targets for the loss
  lengths[0] = max_bits

  # sort in descending order
  lengths = -np.sort(-lengths)

  # chop the bits based on lengths
  for i, sample_length in enumerate(lengths):
    inputs[i, lengths[i]:, ] = 0
    targets[i, lengths[i]:, ] = 0

  return lengths


def evaluate(params, model):
  # evaluate on more bits than training to ensure generalization
  test_loader = DataLoader(
      XORDataset(num_sequences=5000, num_bits=int(params.max_bits * 1.5)), batch_size=500)

  is_correct = np.array([])

  for inputs, targets in test_loader:
    lengths = adjust_lengths(params.vary_lengths, inputs, targets)
    inputs = inputs.to(params.device)
    targets = targets.to(params.device)

    with torch.no_grad():
      logits, predictions = model(inputs, lengths)
      is_correct = np.append(is_correct, ((predictions > 0.5) == (targets > 0.5)))

  accuracy = is_correct.mean()
  return accuracy


if __name__ == '__main__':
  params = get_arguments(ModelParams)
  train(params)
