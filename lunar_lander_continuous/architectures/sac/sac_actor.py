from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SacActor(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.epsilon = 1e-6

        self.tanh = nn.Tanh().to(device=device)
        self.net_fc = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU()
        ).to(device=device)

        self.mean_linear = nn.Linear(in_features=self.hidden_size, out_features=output_size).to(device=device)
        self.log_std_linear = nn.Linear(in_features=self.hidden_size, out_features=output_size).to(device=device)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net_fc(state)

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state=state)
        std = log_std.exp()

        normal = Normal(loc=mean, scale=std)
        z = normal.rsample()

        action = self.tanh(z)

        log_pi = normal.log_prob(value=z) - torch.log(input=(1 - action.pow(2) + self.epsilon))
        log_pi = log_pi.sum(axis=1, keepdim=True)

        return action, log_pi
