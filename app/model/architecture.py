import torch
import torch.nn as nn

class TimeSeriesNet(nn.Module):
    """Простая сеть для прогноза временных рядов"""
    def __init__(self, input_size=12, hidden_size=32, output_size=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)