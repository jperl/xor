import argparse
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


def get_arguments():
  parser = argparse.ArgumentParser()
  register_parser_types(parser, ModelParams)
  arguments = parser.parse_args()
  return arguments


if __name__ == '__main__':
  params = get_arguments()
  train(params)
