import torch.nn as nn


class FOGLSTMModel(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 256,
                 num_layers: int = 2, output_size: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.3,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class FOGRNNModel(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 256,
                 num_layers: int = 2, output_size: int = 3):
        super().__init__()
        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.3,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])