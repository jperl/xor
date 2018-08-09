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
  epochs: int = 10

  # lstm
  hidden_size: int = 1
  learning_rate: float = 1e-1
  num_layers: int = 1


def train(params: ModelParams):
  model = torch.nn.LSTM(
      batch_first=True, input_size=1, hidden_size=params.hidden_size, num_layers=params.num_layers)
  activation = torch.nn.Sigmoid()

  optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate)
  loss_fn = torch.nn.BCEWithLogitsLoss()
  train_loader = DataLoader(XORDataset(), batch_size=params.batch_size, shuffle=True)

  step = 0

  for epoch in range(1, params.epochs):
    for inputs, targets in train_loader:
      # [batch, bits] -> [batch, bits, 1]
      inputs = torch.unsqueeze(inputs, -1)

      # [batch, parity] -> [batch, parity, 1]
      targets = torch.unsqueeze(targets, -1)

      optimizer.zero_grad()

      # reset hidden state per sequence
      h0 = c0 = inputs.new_zeros((params.num_layers, params.batch_size, params.hidden_size))

      logits, _ = model(inputs, (h0, c0))

      # BCEWithLogitsLoss will do the activation (it's more stable)
      loss = loss_fn(logits, targets)
      predictions = activation(logits)

      loss.backward()
      optimizer.step()
      step += 1

      loss_val = loss.item()
      accuracy_val = ((predictions > 0.5) == (targets > 0.5)).type(torch.FloatTensor).mean()

      if step % 500 == 0:
        print(f'epoch {epoch}, step {step}, loss {loss_val:.{4}f}, accuracy {accuracy_val:.{3}f}')

    # evaluate per epoch
    evaluate(model, activation, h0, c0)


def evaluate(model, activation, h0, c0):
  test_loader = DataLoader(XORDataset(train=False), batch_size=params.batch_size)

  prediction_is_correct = np.array([])

  for inputs, targets in test_loader:
    # [batch, bits] -> [batch, bits, 1]
    inputs = torch.unsqueeze(inputs, -1)

    # [batch, parity] -> [batch, parity, 1]
    targets = torch.unsqueeze(targets, -1)

    with torch.no_grad():
      logits, _ = model(inputs, (h0, c0))
      predictions = activation(logits)
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
