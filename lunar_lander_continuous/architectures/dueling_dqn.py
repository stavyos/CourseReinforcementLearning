import numpy as np
import torch
from gym.envs.box2d.lunar_lander import LunarLanderContinuous
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DuelingDQN(nn.Module):
    def __init__(self, env: LunarLanderContinuous, hidden_size: int, out_features: int, optimizer_lr: float):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))

        self.hidden_size = hidden_size
        self.out_features = out_features
        self.optimizer_lr = optimizer_lr

        self.fc = nn.Linear(in_features=in_features, out_features=self.hidden_size).to(device=device)
        self.tanh = nn.Tanh().to(device=device)
        self.fc_value = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size).to(device=device)
        self.fc_adv = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size).to(device=device)

        self.value = nn.Linear(in_features=self.hidden_size, out_features=1).to(device=device)
        self.adv = nn.Linear(in_features=self.hidden_size, out_features=out_features).to(device=device)

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=self.optimizer_lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc_result = self.tanh(self.fc(x))
        value = self.tanh(self.fc_value(fc_result))
        advantage = self.tanh(self.fc_adv(fc_result))

        value = self.value(value)
        advantage = self.adv(advantage)

        q_value = value + (advantage - torch.mean(advantage, dim=1, keepdim=True))

        return q_value

    def get_action(self, state: np.ndarray) -> int:
        state = np.expand_dims(state, axis=0)

        with torch.no_grad():  # TODO: With this "with" or there is no need for this...?
            q_value = self.forward(torch.as_tensor(state))
            action_index = torch.argmax(q_value, dim=1)
            return action_index.detach().item()
