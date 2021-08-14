import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Critic(nn.Module):
    def __init__(self, state_input_size: int, action_input_size: int, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size

        self.net = nn.Sequential(
            nn.Linear(in_features=state_input_size + action_input_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=1)
        ).to(device=device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat(tensors=[state, action], dim=1))
