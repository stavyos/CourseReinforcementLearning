import gym
import torch
import torch.nn as nn

from lunar_lander_continuous.architectures.sac.sac_agent import SacAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SacActor_tmp(nn.Module):
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


class Critic_tmp(nn.Module):
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


env = gym.make('LunarLanderContinuous-v2')
agent = SacAgent(env=env, hidden_size=256, delay_step=2)
agent.load_model('models//SAC//', episode=240, avg_100=208)

actor_tmp = SacActor_tmp(input_size=agent.state_input_size,
                         output_size=agent.action_input_size,
                         hidden_size=agent.hidden_size)
critic_tmp1 = Critic_tmp(state_input_size=agent.state_input_size,
                         action_input_size=agent.action_input_size,
                         hidden_size=agent.hidden_size)

critic_tmp2 = Critic_tmp(state_input_size=agent.state_input_size,
                         action_input_size=agent.action_input_size,
                         hidden_size=agent.hidden_size)

critic_params1 = [agent.q_net_1_online.linear1.weight,
                  agent.q_net_1_online.linear1.bias,
                  agent.q_net_1_online.linear2.weight,
                  agent.q_net_1_online.linear2.bias,
                  agent.q_net_1_online.linear3.weight,
                  agent.q_net_1_online.linear3.bias]

critic_params2 = [agent.q_net_2_online.linear1.weight,
                  agent.q_net_2_online.linear1.bias,
                  agent.q_net_2_online.linear2.weight,
                  agent.q_net_2_online.linear2.bias,
                  agent.q_net_2_online.linear3.weight,
                  agent.q_net_2_online.linear3.bias]

for dest_params, params in zip(critic_tmp1.net.parameters(), critic_params1):
    dest_params.data.copy_(params)

for dest_params, params in zip(critic_tmp2.net.parameters(), critic_params2):
    dest_params.data.copy_(params)

# torch.save(actor_tmp.state_dict(), 'models/SAC/actor_240_208.pth')
torch.save(critic_tmp1.state_dict(), 'models/SAC/q1_240_208.pth')
torch.save(critic_tmp2.state_dict(), 'models/SAC/q2_240_208.pth')

sds = 4
