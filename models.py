import torch.nn as nn


class FOGLSTMModel(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 256, num_layers: int = 2, output_size: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class FOGRNNModel(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 256, num_layers: int = 2, output_size: int = 3):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class FOG1DCNNModel(nn.Module):
    def __init__(self, input_channels: int = 3, output_size: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, output_size)

    def forward(self, x):
        return self.fc(self.encoder(x.permute(0, 2, 1)).squeeze(-1))