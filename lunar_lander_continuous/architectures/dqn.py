import numpy as np
import torch
from gym.envs.box2d.lunar_lander import LunarLanderContinuous
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, env: LunarLanderContinuous, hidden_size: int, out_features: int, optimizer_lr: float):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))

        self.hidden_size = hidden_size
        self.out_features = out_features
        self.optimizer_lr = optimizer_lr

        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=self.hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=self.hidden_size, out_features=out_features)
        ).to(device=device)

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=self.optimizer_lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_action(self, state: np.ndarray) -> int:
        state_t = torch.FloatTensor(np.expand_dims(state, axis=0))
        q_value = self(state_t)

        max_q_index = torch.argmax(q_value, dim=1)[0]
        action = max_q_index.detach().item()

        return action
