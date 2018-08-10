import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import NamedTuple
from xor_dataset import XORDataset
from utils import register_parser_types


class ModelParams(NamedTuple):
  # train loop
  batch_size: int = 8
  device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  epochs: int = 15
  resume_path: str = None

  # lstm
  hidden_size: int = 2
  lr: float = 5e-1
  momentum: float = 0.9
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

  optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum)
  loss_fn = torch.nn.BCEWithLogitsLoss()
  train_loader = DataLoader(XORDataset(), batch_size=params.batch_size, shuffle=True)
  test_loader = DataLoader(XORDataset(train=False), batch_size=params.batch_size)

  step = 0
  epoch = 1

  if params.resume_path:
    step, epoch = resume_train_state(params.resume_path, model, optimizer)

  for epoch in range(epoch, params.epochs):
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

      if step % 500 == 0:
        print(f'epoch {epoch}, step {step}, loss {loss.item():.{4}f}, accuracy {accuracy:.{3}f}')

    # evaluate per epoch
    evaluate(model, test_loader)
    save_train_state(step, epoch, model, optimizer)


def resume_train_state(path, model, optimizer):
  state = torch.load(path)
  model.load_state_dict(state['model'])
  optimizer.load_state_dict(state['optimizer'])
  return state['step'], state['epoch']


def save_train_state(step, epoch, model, optimizer):
  state = {
      'epoch': epoch + 1,
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'step': step
  }
  torch.save(state, f'./data/epoch_{epoch}.pt')


def evaluate(model, loader):
  is_correct = np.array([])

  for inputs, targets in loader:
    inputs = inputs.to(params.device)
    targets = targets.to(params.device)
    with torch.no_grad():
      logits, predictions = model(inputs)
      is_correct = np.append(is_correct, ((predictions > 0.5) == (targets > 0.5)))

  accuracy = is_correct.mean()
  print(f'test accuracy {accuracy:.{3}f}')


def get_arguments():
  parser = argparse.ArgumentParser()
  register_parser_types(parser, ModelParams)
  arguments = parser.parse_args()
  return arguments


if __name__ == '__main__':
  params = get_arguments()
  train(params)
