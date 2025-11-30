import torch
from torch import nn


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
    def __init__(
        self, input_size=512, reservoir_size=1000, num_classes=101, spectral_radius=0.9
    ):
        super().__init__()
        self.reservoir_size = reservoir_size

        # Fixed random reservoir weights
        self.W_in = nn.Linear(input_size, reservoir_size, bias=False)
        self.W_res = nn.Parameter(
            torch.randn(reservoir_size, reservoir_size), requires_grad=False
        )

        # Freeze input weights too (ESN principle)
        for param in self.W_in.parameters():
            param.requires_grad = False

        # Scale reservoir weights by spectral radius
        with torch.no_grad():
            eigenvals = torch.linalg.eigvals(self.W_res).real
            self.W_res *= spectral_radius / torch.max(eigenvals)

        # Only train the readout layer
        self.W_readout = nn.Linear(reservoir_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.reservoir_size, device=x.device)

        # Run through reservoir
        for t in range(seq_len):
            u = self.W_in(x[:, t, :])
            h = torch.tanh(u + h @ self.W_res)

        return self.W_readout(h)


class SimpleESNWithNonLinearReadout(nn.Module):
    def __init__(
        self,
        input_size=512,
        reservoir_size=1000,
        num_classes=101,
        spectral_radius=0.9,
        hidden_readout_size=256,
    ):
        super().__init__()
        self.reservoir_size = reservoir_size

        self.W_in = nn.Linear(input_size, reservoir_size, bias=False)
        self.W_res = nn.Parameter(
            torch.randn(reservoir_size, reservoir_size), requires_grad=False
        )

        for param in self.W_in.parameters():
            param.requires_grad = False

        with torch.no_grad():
            eigenvals = torch.linalg.eigvals(self.W_res).real
            self.W_res *= spectral_radius / torch.max(eigenvals)

        self.W_readout = nn.Sequential(
            nn.Linear(reservoir_size, hidden_readout_size),
            nn.ReLU(),
            nn.Linear(hidden_readout_size, num_classes),
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.reservoir_size, device=x.device)

        for t in range(seq_len):
            u = self.W_in(x[:, t, :])
            h = torch.tanh(u + h @ self.W_res)

        return self.W_readout(h)


class DeepESN(nn.Module):
    def __init__(
        self,
        input_size=512,
        reservoir_size=1000,
        num_layers=3,
        num_classes=101,
        spectral_radius=0.9,
    ):
        super().__init__()
        self.reservoir_size = reservoir_size
        self.num_layers = num_layers

        # Input layer to first reservoir
        self.W_in = nn.Linear(input_size, reservoir_size, bias=False)

        # Freeze input weights (ESN principle)
        for param in self.W_in.parameters():
            param.requires_grad = False

        # Multiple reservoir layers
        self.reservoirs = nn.ModuleList()
        for _ in range(num_layers):
            w_res = nn.Parameter(
                torch.randn(reservoir_size, reservoir_size), requires_grad=False
            )
            # Scale by spectral radius
            with torch.no_grad():
                eigenvals = torch.linalg.eigvals(w_res).real
                w_res *= spectral_radius / torch.max(eigenvals)
            self.reservoirs.append(nn.ParameterDict({"W_res": w_res}))

        # Layer-to-layer connections (optional - can be zero for independent layers)
        self.layer_connections = nn.ModuleList()
        for _ in range(num_layers - 1):
            layer_conn = nn.Linear(reservoir_size, reservoir_size, bias=False)
            # Freeze layer connections too
            for param in layer_conn.parameters():
                param.requires_grad = False
            self.layer_connections.append(layer_conn)

        # Readout from all layers (concatenated)
        self.readout = nn.Linear(reservoir_size * num_layers, num_classes)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Initialize all reservoir states
        h_layers = [
            torch.zeros(batch_size, self.reservoir_size, device=x.device)
            for _ in range(self.num_layers)
        ]

        # Process sequence through all reservoir layers
        for t in range(seq_len):
            # First layer gets input
            u = self.W_in(x[:, t, :])
            h_layers[0] = torch.tanh(u + h_layers[0] @ self.reservoirs[0]["W_res"])

            # Subsequent layers get input from previous layer + their own recurrence
            for i in range(1, self.num_layers):
                layer_input = self.layer_connections[i - 1](h_layers[i - 1])
                h_layers[i] = torch.tanh(
                    layer_input + h_layers[i] @ self.reservoirs[i]["W_res"]
                )

        # Concatenate final states from all layers
        final_state = torch.cat(h_layers, dim=1)
        return self.readout(final_state)


class DeepRNN(nn.Module):
    def __init__(
        self,
        input_size=512,
        hidden_size=128,
        num_layers=3,
        num_classes=101,
        dropout=0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Multi-layer LSTM with dropout
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Optional: Additional fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.rnn(x)
        # Take last timestep and pass through FC layers
        out = self.fc_layers(out[:, -1, :])
        return out
