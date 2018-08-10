import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import NamedTuple
from xor_dataset import XORDataset
from utils import register_parser_types


class ModelParams(NamedTuple):
  # train loop
  batch_size: int = 32
  device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  epochs: int = 5

  # lstm
  hidden_size: int = 2
  learning_rate: float = 1
  num_layers: int = 1


class LSTM(torch.nn.Module):
  def __init__(self, params: ModelParams):
    super().__init__()

    self._params = params

    # dropout = 0 if config.n_layers == 1 else config.dp_ratio
    self.lstm = torch.nn.LSTM(
        batch_first=True,
        input_size=1,
        hidden_size=params.hidden_size,
        num_layers=params.num_layers)

    # The linear layer that maps from hidden state space to the logits
    self.hidden2logits = torch.nn.Linear(params.hidden_size, 1)

    self.activation = torch.nn.Sigmoid()

  def forward(self, inputs):
    batch_size = inputs.size()[0]

    # reset hidden state per sequence
    h0 = c0 = inputs.new_zeros((params.num_layers, params.batch_size, params.hidden_size))

    lstm_out, _ = self.lstm(inputs, (h0, c0))
    logits = self.hidden2logits(lstm_out)

    predictions = self.activation(logits)

    return logits, predictions


def train(params: ModelParams):
  model = LSTM(params).to(params.device)

  optimizer = torch.optim.Adam(model.parameters())  #, lr=params.learning_rate)
  loss_fn = torch.nn.BCEWithLogitsLoss()
  train_loader = DataLoader(XORDataset(), batch_size=params.batch_size, shuffle=True)

  step = 0

  for epoch in range(1, params.epochs):
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

      loss_val = loss.item()
      accuracy_val = ((predictions > 0.5) == (targets > 0.5)).type(torch.FloatTensor).mean()

      if step % 500 == 0:
        print(f'epoch {epoch}, step {step}, loss {loss_val:.{4}f}, accuracy {accuracy_val:.{3}f}')

    # evaluate per epoch
    evaluate(model)


def evaluate(model):
  test_loader = DataLoader(XORDataset(train=False), batch_size=params.batch_size)

  prediction_is_correct = np.array([])

  for inputs, targets in test_loader:
    inputs = inputs.to(params.device)
    targets = targets.to(params.device)
    with torch.no_grad():
      logits, predictions = model(inputs)
      prediction_is_correct = np.append(prediction_is_correct,
                                        ((predictions > 0.5) == (targets > 0.5)))

  accuracy_val = prediction_is_correct.mean()
  print(f'test accuracy {accuracy_val:.{3}f}')


def get_arguments():
  parser = argparse.ArgumentParser()
  register_parser_types(parser, ModelParams)
  arguments = parser.parse_args()
  return arguments


if __name__ == '__main__':
  params = get_arguments()
  train(params)
