import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from gym.envs.box2d import LunarLanderContinuous

from lunar_lander_continuous.architectures.sac.critic import Critic
from lunar_lander_continuous.architectures.sac.sac_actor import SacActor
from lunar_lander_continuous.replay_memory import ReplayMemory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SacAgent:
    def __init__(self, env: LunarLanderContinuous, hidden_size: int, delay_step: int):
        self.state_input_size = env.observation_space.shape[0]
        self.action_input_size = env.action_space.shape[0]

        self.hidden_size = hidden_size

        self.batch_size = 128
        self.tau = 0.005
        self.gamma = 0.99
        self.q_lr = 3e-4
        self.actor_lr = 3e-4
        self.alpha_lr = 3e-3

        self.update_step = 0
        self.delay_step = delay_step

        self.action_range = [env.action_space.low, env.action_space.high]

        self.replay_memory = ReplayMemory(transition_size=[8, 2, 1, 1, 8], maxlen=1000000)

        # entropy temperature
        self.alpha = 0.2
        self.target_entropy = -torch.prod(input=torch.Tensor(env.action_space.shape)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = optim.Adam(params=[self.log_alpha], lr=self.alpha_lr)

        self.actor = SacActor(input_size=self.state_input_size,
                              output_size=self.action_input_size,
                              hidden_size=self.hidden_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.q_net_1_online = Critic(state_input_size=self.state_input_size,
                                     action_input_size=self.action_input_size,
                                     hidden_size=self.hidden_size)
        self.q_net_1_target = Critic(state_input_size=self.state_input_size,
                                     action_input_size=self.action_input_size,
                                     hidden_size=self.hidden_size)
        self.q_net_1_target.load_state_dict(state_dict=self.q_net_1_online.state_dict())
        self.q_net_1_optimizer = optim.Adam(params=self.q_net_1_online.parameters(), lr=self.q_lr)

        self.q_net_2_online = Critic(state_input_size=self.state_input_size,
                                     action_input_size=self.action_input_size,
                                     hidden_size=self.hidden_size)
        self.q_net_2_target = Critic(state_input_size=self.state_input_size,
                                     action_input_size=self.action_input_size,
                                     hidden_size=self.hidden_size)
        self.q_net_2_target.load_state_dict(state_dict=self.q_net_2_online.state_dict())
        self.q_net_2_optimizer = optim.Adam(params=self.q_net_2_online.parameters(), lr=self.q_lr)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.FloatTensor(state).unsqueeze(0)

        action, log_pi = self.actor.sample(state)
        action = action.detach().squeeze(0).numpy()

        return action

    def save(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray, fail: bool):
        self.replay_memory.push(state=state, action=action, reward=reward, done=fail, next_state=new_state)

    def update(self):
        if self.replay_memory.buff_pointer < self.batch_size:
            return

        states, actions, rewards, fails, next_states = self.replay_memory.get_batch(batch_size=self.batch_size)
        not_fails = fails == 0

        next_actions, next_log_pi = self.actor.sample(state=next_states)

        next_q_1 = self.q_net_1_target(next_states, next_actions)
        next_q_2 = self.q_net_2_target(next_states, next_actions)

        next_q_target = torch.min(next_q_1, next_q_2) - self.alpha * next_log_pi
        expected_q = rewards + not_fails * self.gamma * next_q_target

        curr_q_1 = self.q_net_1_online.forward(states, actions)
        curr_q_2 = self.q_net_2_online.forward(states, actions)

        q1_loss = F.mse_loss(curr_q_1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q_2, expected_q.detach())

        self.q_net_1_optimizer.zero_grad()
        q1_loss.backward()
        self.q_net_1_optimizer.step()

        self.q_net_2_optimizer.zero_grad()
        q2_loss.backward()
        self.q_net_2_optimizer.step()

        # delayed update for policy network and target q networks
        new_actions, log_pi = self.actor.sample(state=states)
        if self.update_step % self.delay_step == 0:
            min_q = torch.min(self.q_net_1_online.forward(states, new_actions),
                              self.q_net_2_online.forward(states, new_actions))
            actor_loss = (self.alpha * log_pi - min_q).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # target networks
            for target_param, param in zip(self.q_net_1_target.parameters(), self.q_net_1_online.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            for target_param, param in zip(self.q_net_2_target.parameters(), self.q_net_2_online.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # update temperature
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        self.update_step += 1

    def save_model(self, data_dir: str, episode: int, avg_100: int):
        avg_100 = int(avg_100)

        torch.save(self.actor.state_dict(),
                   os.path.join(data_dir, f'actor_{episode}_{avg_100}.pth'))
        torch.save(self.q_net_1_online.state_dict(),
                   os.path.join(data_dir, f'q1_{episode}_{avg_100}.pth'))
        torch.save(self.q_net_2_online.state_dict(),
                   os.path.join(data_dir, f'q2_{episode}_{avg_100}.pth'))

    def load_model(self, data_dir: str, episode: int, avg_100: int):
        actor_dir = os.path.join(data_dir, f'actor_{episode}_{avg_100}.pth')
        self.actor.load_state_dict(torch.load(actor_dir))

        q_net_1_dir = os.path.join(data_dir, f'q1_{episode}_{avg_100}.pth')
        self.q_net_1_online.load_state_dict(torch.load(q_net_1_dir))
        self.q_net_1_target.load_state_dict(self.q_net_1_online.state_dict())

        q_net_2_dir = os.path.join(data_dir, f'q2_{episode}_{avg_100}.pth')
        self.q_net_2_online.load_state_dict(torch.load(q_net_2_dir))
        self.q_net_2_target.load_state_dict(self.q_net_2_online.state_dict())
