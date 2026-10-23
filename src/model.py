import torch
import torch.nn as nn

class EngineANN(nn.Module):
    def __init__(self, input_dim=7, output_dim=8):
        super(EngineANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)  