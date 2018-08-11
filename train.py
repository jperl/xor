import numpy as np
import torch
from torch.nn.utils import rnn as rnn_utils
from torch.utils.data import DataLoader
from typing import NamedTuple
from xor_dataset import XORDataset
from utils import get_arguments


class ModelParams(NamedTuple):
  # train loop
  batch_size: int = 8
  batch_size_test: int = 256
  device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  epochs: int = 1

  # lstm
  hidden_size: int = 2
  lr: float = 5e-1
  momentum: float = 0.99
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

  def forward(self, inputs):
    batch_size = inputs.size()[0]
    num_bits = inputs.size()[1]

    # pack the inputs
    lengths = torch.ones(batch_size, dtype=torch.int) * num_bits
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

  # evaluate on separate data from training
  train_dataset, test_dataset = XORDataset(), XORDataset()
  train_loader = DataLoader(train_dataset, batch_size=params.batch_size)
  test_loader = DataLoader(test_dataset, batch_size=params.batch_size_test)

  step = 0

  for epoch in range(1, params.epochs + 1):
    for inputs, targets in train_loader:
      inputs = inputs.to(params.device)
      targets = targets.to(params.device)
      optimizer.zero_grad()

      logits, predictions = model(inputs)

      # BCEWithLogitsLoss will do the activation (it's more stable)
      loss = loss_fn(logits, targets)

      loss.backward()
      optimizer.step()
      step += 1

      accuracy = ((predictions > 0.5) == (targets > 0.5)).type(torch.FloatTensor).mean()

      if step % 250 == 0:
        print(f'epoch {epoch}, step {step}, loss {loss.item():.{4}f}, accuracy {accuracy:.{3}f}')

      if step % 1000 == 0:
        test_accuracy = evaluate(model, test_loader)
        print(f'test accuracy {test_accuracy:.{3}f}')
        if test_accuracy == 1.0:
          # stop early
          break


def evaluate(model, loader):
  is_correct = np.array([])

  for inputs, targets in loader:
    inputs = inputs.to(params.device)
    targets = targets.to(params.device)
    with torch.no_grad():
      logits, predictions = model(inputs)
      is_correct = np.append(is_correct, ((predictions > 0.5) == (targets > 0.5)))

  accuracy = is_correct.mean()
  return accuracy


if __name__ == '__main__':
  params = get_arguments(ModelParams)
  train(params)
