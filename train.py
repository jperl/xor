import torch
from torch.utils.data import DataLoader
from xor_dataset import XORDataset

BATCH_SIZE = 32

model = torch.nn.LSTM(batch_first=True, input_size=1, hidden_size=1, num_layers=1)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-03)
loss_fn = torch.nn.BCEWithLogitsLoss()
train_loader = DataLoader(XORDataset(), batch_size=BATCH_SIZE, shuffle=True)

step = 0

for inputs, targets in train_loader:
  # [batch, bits] -> [batch, bits, 1]
  inputs = torch.unsqueeze(inputs, -1)

  # [1] -> [1, 1]
  targets = torch.unsqueeze(targets, -1)

  optimizer.zero_grad()

  # reset hidden state per sequence
  # state is (num_cells, batch_size, hidden_size)
  state_shape = 1, BATCH_SIZE, 1
  h0 = c0 = inputs.new_zeros(state_shape)

  final_outputs, _ = model(inputs, (h0, c0))

  # select the last prediction
  # XXX we should calculate parity per bit in the lstm
  loss = loss_fn(final_outputs[:, -1], targets)

  loss.backward()
  optimizer.step()
  step += 1

  loss_val = loss.item()
  if step % 500 == 0:
    print(f'LOSS step {step}: {loss_val}')
