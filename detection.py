import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes):
        super(CNN_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_hidden_state = lstm_out[:, -1, :]
        out = self.fc(final_hidden_state)
        return out
