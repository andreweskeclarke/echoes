import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, num_classes=101, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Take last timestep
        return out


class SimpleESN(nn.Module):
    def __init__(self, input_size=512, reservoir_size=1000, num_classes=101, spectral_radius=0.9):
        super().__init__()
        self.reservoir_size = reservoir_size

        # Fixed random reservoir weights
        self.W_in = nn.Linear(input_size, reservoir_size, bias=False)
        self.W_res = nn.Parameter(torch.randn(reservoir_size, reservoir_size), requires_grad=False)

        # Scale reservoir weights by spectral radius
        with torch.no_grad():
            eigenvals = torch.linalg.eigvals(self.W_res).real
            self.W_res *= spectral_radius / torch.max(eigenvals)

        # Only train the readout layer
        self.readout = nn.Linear(reservoir_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.reservoir_size, device=x.device)

        # Run through reservoir
        for t in range(seq_len):
            u = self.W_in(x[:, t, :])
            h = torch.tanh(u + h @ self.W_res)

        return self.readout(h)