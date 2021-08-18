import json
import math
import os
import random
from collections import deque
from typing import Tuple

import gym
import numpy as np
import pandas as pd
import torch
from gym.envs.box2d.lunar_lander import LunarLanderContinuous
from torch import nn
from torch.nn import functional

from architectures.dqn import DQN
from architectures.sac.sac_agent import SacAgent
from replay_memory import ReplayMemory

EPISODES = 1000
MAX_STEPS_PER_EPISODE = 500
GAMMA = 0.99
BATCH_SIZE = 128
BUFFER_SIZE = 1000000
MIN_MEMORY_REPLAY_SIZE = 50000
EPSILON_START = 1
EPSILON_END = 0.1
EPSILON_DECAY = 100000
OPTIMIZER_LR = 5e-4
TARGET_UPDATE_FREQ = 2000
TRANSITION_SIZE = [8, 1, 1, 1, 8]  # state, action, reward, done, new_state

# region Actions Section
# Example, ACTIONS_QUANTITATIVE = [-1, -0.5, 0, 0.5, 1] where ACTIONS_QUANTIZATION_LENGTH = 5
ACTIONS_QUANTIZATION_LENGTH = 9

ACTIONS_QUANTITATIVE = np.round(np.linspace(-1, 1, ACTIONS_QUANTIZATION_LENGTH), 3)[np.newaxis, :]
ACTIONS = {}
ACTIONS_REVERSE = {}


def get_parameters() -> dict:
    d = {
        'EPISODES': EPISODES,
        'MAX_STEPS_PER_EPISODE': MAX_STEPS_PER_EPISODE,
        'GAMMA': GAMMA,
        'BATCH_SIZE': BATCH_SIZE,
        'BUFFER_SIZE': BUFFER_SIZE,
        'MIN_MEMORY_REPLAY_SIZE': MIN_MEMORY_REPLAY_SIZE,
        'EPSILON_START': EPSILON_START,
        'EPSILON_END': EPSILON_END,
        'EPSILON_DECAY': EPSILON_DECAY,
        'OPTIMIZER_LR': OPTIMIZER_LR,
        'TARGET_UPDATE_FREQ': TARGET_UPDATE_FREQ,
        'TRANSITION_SIZE': TRANSITION_SIZE,
        'ACTIONS_QUANTIZATION_LENGTH': ACTIONS_QUANTIZATION_LENGTH
    }

    return d


def create_actions():
    global ACTIONS_QUANTITATIVE
    ACTIONS_QUANTITATIVE = np.round(np.linspace(-1, 1, ACTIONS_QUANTIZATION_LENGTH), 3)[np.newaxis, :]

    c = 0
    for i in ACTIONS_QUANTITATIVE[0, :]:
        for j in ACTIONS_QUANTITATIVE[0, :]:
            ACTIONS.update({(i, j): c})
            ACTIONS_REVERSE.update({c: (i, j)})
            c += 1


def get_action_1d(action_2d: Tuple[float, float]) -> int:
    return ACTIONS.get(action_2d)


def get_action_2d(action_1d: int) -> np.ndarray:
    return np.array(ACTIONS_REVERSE.get(action_1d))


def round_actions(action: np.ndarray) -> np.ndarray:
    _action = action.copy()
    _action = np.expand_dims(_action, axis=1)

    closest_indices = np.argmin(np.abs(np.subtract(_action, ACTIONS_QUANTITATIVE)), axis=1)
    actions_rounded = ACTIONS_QUANTITATIVE[:, closest_indices]

    return actions_rounded.flatten()


# endregion


def initialize_replay_memory(env: LunarLanderContinuous, replay_memory: ReplayMemory):
    state = env.reset()
    for _ in range(MIN_MEMORY_REPLAY_SIZE):
        action = env.action_space.sample()
        action = round_actions(action=action)

        new_state, reward, done, _ = env.step(action=action)

        action = get_action_1d(action_2d=tuple(action))
        replay_memory.push(state=state, action=action, reward=reward, done=done, next_state=new_state)
        state = new_state

        if done:
            state = env.reset()


def train_dqn_dueling_dqn(env: LunarLanderContinuous,
                          online_net: nn.Module,
                          target_net: nn.Module,
                          replay_memory: ReplayMemory,
                          reward_buffer: deque) -> pd.DataFrame:
    cols = ['Episode', 'Reward', 'Steps', 'AvgReward', 'TotalSteps', 'Epsilon']
    df = pd.DataFrame(columns=cols)

    total_steps = 0
    for episode in range(1, EPISODES + 1):
        episode_reward = 0.0
        state = env.reset()

        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            total_steps += 1
            epsilon = np.interp(total_steps, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

            if random.random() <= epsilon and False:
                action = env.action_space.sample()
                action = round_actions(action=action)
            else:
                action = online_net.get_action(state=state)
                action = get_action_2d(action_1d=action)

            new_state, reward, done, _ = env.step(action=action)
            episode_reward += reward

            # if episode % 100 == 0:
            #     env.render()

            action = get_action_1d(action_2d=tuple(action))
            replay_memory.push(state=state, action=action, reward=reward, done=done, next_state=new_state)
            state = new_state

            # Start Gradient Step
            states_t, actions_t, rewards_t, dones_t, new_states_t = replay_memory.get_batch(batch_size=BATCH_SIZE)

            # Compute Targets
            target_q_values = target_net(new_states_t)
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
            targets = rewards_t + GAMMA * (1 - dones_t) * max_target_q_values

            # Compute Loss
            q_values = online_net(states_t)
            action_q_values = torch.gather(input=q_values, dim=1, index=torch.LongTensor(actions_t.detach().numpy()))

            # https://pytorch.org/docs/1.9.0/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss
            # Similar to HuberLoss.
            loss = functional.smooth_l1_loss(input=action_q_values, target=targets)

            # Gradient Descent Step
            online_net.optimizer.zero_grad()
            loss.backward()
            online_net.optimizer.step()

            # Update Target Network
            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(online_net.state_dict())

            if done:
                break

        reward_buffer.append(episode_reward)

        avg_reward = np.mean(reward_buffer)
        print('Avg reward', np.round(avg_reward, 1),
              'Total Steps', total_steps, end='\t\t')
        # noinspection PyUnboundLocalVariable
        print('Episode', episode,
              'Episode steps', step,
              'Episode reward', round(episode_reward, 1),
              'Epsilon', epsilon)

        df = df.append(
            {
                'Episode': episode,
                'Reward': episode_reward,
                'Steps': step,
                'AvgReward': avg_reward,
                'TotalSteps': total_steps,
                'Epsilon': epsilon
            },
            ignore_index=True)

        if len(reward_buffer) >= 100 and avg_reward >= 200:
            print('Solved')
            break

    return df


def start_dqn_dueling_dqn(network_architecture: type, dir_experiment: str):
    replay_memory = ReplayMemory(transition_size=TRANSITION_SIZE, maxlen=BUFFER_SIZE)
    reward_buffer = deque(maxlen=100)

    env: LunarLanderContinuous = gym.make('LunarLanderContinuous-v2').unwrapped

    out_features = ACTIONS_QUANTIZATION_LENGTH ** 2
    online_net = network_architecture(env=env, hidden_size=128, out_features=out_features, optimizer_lr=OPTIMIZER_LR)
    target_net = network_architecture(env=env, hidden_size=128, out_features=out_features, optimizer_lr=OPTIMIZER_LR)

    target_net.load_state_dict(online_net.state_dict())

    initialize_replay_memory(env=env, replay_memory=replay_memory)

    df_result = train_dqn_dueling_dqn(env=env,
                                      online_net=online_net,
                                      target_net=target_net,
                                      replay_memory=replay_memory,
                                      reward_buffer=reward_buffer)

    df_result.to_csv(os.path.join(dir_experiment, 'logging.csv'), sep=',', index=False)

    with open(os.path.join(dir_experiment, 'parameters.json'), 'w') as f:
        json.dump(get_parameters(), f)

    torch.save(online_net.state_dict(), os.path.join(dir_experiment, 'model.pth'))


def main_dqn_dueling_dqn():
    global ACTIONS_QUANTIZATION_LENGTH

    # actions_length = [5, 7, 9, 11, 13, 15]
    actions_length = [11, 11, 15]

    def create_folder(_folder: str):
        if not os.path.exists(_folder):
            os.mkdir(_folder)

    folder = 'models'
    create_folder(_folder=folder)

    folder = os.path.join(folder, 'architecture')
    create_folder(_folder=folder)

    folder = os.path.join(folder, 'layers_333')
    create_folder(_folder=folder)

    for action_length in actions_length:
        ACTIONS_QUANTIZATION_LENGTH = action_length
        create_actions()

        folder_model = os.path.join(folder, f'8_128_tanh_128_tanh_{action_length ** 2}//')
        create_folder(_folder=folder_model)

        start_dqn_dueling_dqn(network_architecture=DQN, dir_experiment=folder_model)
        break


def run_one_episode_sac(agent: SacAgent, env: gym.Env, max_steps: int, test: bool = False) -> Tuple[float, int, bool]:
    old_state = env.reset()

    if test:
        env.render('rgb_array')

    episode_reward = 0
    fail = False
    step = 1

    agent.total_time = 0
    agent.total_time2 = 0
    while step < max_steps:
        action = agent.get_action(state=old_state)

        new_state, reward, done, info = env.step(action=action)

        if test:
            env.render('rgb_array')

        fail = done if step + 1 < max_steps else False

        agent.save(state=old_state, action=action, reward=reward, new_state=new_state, fail=fail)
        agent.update()

        old_state = new_state
        episode_reward += reward
        if done:
            break

        step += 1
    return episode_reward, step, fail


def train_sac(agent: SacAgent, env: gym.Env, num_of_episodes: int, max_steps: int):
    reward_buffer = deque(maxlen=100)

    cols = ['Episode', 'Reward', 'Steps', 'AvgReward', 'TotalSteps']
    df_result = pd.DataFrame(columns=cols)
    total_steps = 0
    for episode in range(1, num_of_episodes + 1):
        reward, step, fail = run_one_episode_sac(agent=agent, env=env, max_steps=max_steps)

        reward_buffer.append(reward)
        total_steps += step

        print('Episode', episode,
              'Reward', reward,
              'Steps', step,
              'Avg Reward', np.round(np.mean(reward_buffer), 1),
              'Total Steps', total_steps)

        df_result = df_result.append(
            {
                'Episode': episode,
                'Reward': reward,
                'Steps': step,
                'AvgReward': np.mean(reward_buffer),
                'TotalSteps': total_steps
            },
            ignore_index=True)

        if len(reward_buffer) >= 10 and np.mean(reward_buffer) >= -250:
            agent.save_model(data_dir='models', episode=episode, avg_100=np.round(np.mean(reward_buffer), 1))
            df_result.to_csv(os.path.join('', 'logging.csv'), sep=',', index=False)


def main_sac():
    env = gym.make('LunarLanderContinuous-v2')
    agent = SacAgent(env=env, hidden_size=256, delay_step=2)

    train_sac(agent=agent, env=env, num_of_episodes=EPISODES, max_steps=MAX_STEPS_PER_EPISODE)


def testing_all_models(layers: int):
    global ACTIONS_QUANTIZATION_LENGTH

    env: LunarLanderContinuous = gym.make('LunarLanderContinuous-v2').unwrapped

    _path = f'models//DQN//layers_{layers}//'

    cols = ['Episode', 'Reward', 'Steps', 'AvgReward']
    max_episodes = 1000
    for network in os.listdir(_path):
        features = int(str.split(network, '_')[-1])
        ACTIONS_QUANTIZATION_LENGTH = int(math.sqrt(features))
        create_actions()
        print(f'------{features}------')

        online_net = DQN(env=env, hidden_size=128, out_features=features, optimizer_lr=OPTIMIZER_LR)
        online_net.load_state_dict(torch.load(os.path.join(_path, network, 'model.pth')))

        df = pd.DataFrame(columns=cols)
        reward_buffer = deque(maxlen=max_episodes)
        episode_reward, episode_steps = 0, 0
        state = env.reset()
        while True:
            episode_steps += 1
            action = online_net.get_action(state=state)
            action = get_action_2d(action_1d=action)

            state, reward, done, _ = env.step(action=action)
            episode_reward += reward
            # env.render()
            if done or episode_steps >= 500:
                reward_buffer.append(episode_reward)

                avg_reward = np.mean(reward_buffer)
                print('Episode', len(reward_buffer),
                      'Avg reward', np.round(avg_reward, 1),
                      'Episode steps', episode_steps,
                      'Episode reward', round(episode_reward, 1))

                df = df.append(
                    {
                        'Episode': len(reward_buffer),
                        'Reward': episode_reward,
                        'Steps': episode_steps,
                        'AvgReward': avg_reward,
                    },
                    ignore_index=True)

                episode_reward = 0
                episode_steps = 0
                env.reset()

            if len(reward_buffer) == max_episodes:
                df.to_csv(f'testing_{network}.csv')
                print('Done!', np.round(np.mean(reward_buffer), 1))
                break


def testing_sac():
    env: LunarLanderContinuous = gym.make('LunarLanderContinuous-v2').unwrapped

    agent = SacAgent(env=env, hidden_size=256, delay_step=2)
    agent.load_model('models//SAC//', episode=820, avg_100=241)

    cols = ['Episode', 'Reward', 'Steps', 'AvgReward']
    df = pd.DataFrame(columns=cols)

    max_episodes = 1000
    reward_buffer = deque(maxlen=max_episodes)

    episode_reward = 0
    episode_steps = 0
    state = env.reset()
    while True:
        episode_steps += 1
        action = agent.get_action(state=state)

        state, reward, done, _ = env.step(action=action)
        episode_reward += reward
        # env.render()
        if done or episode_steps >= 500:
            reward_buffer.append(episode_reward)

            avg_reward = np.mean(reward_buffer)
            print('Episode', len(reward_buffer),
                  'Avg reward', np.round(avg_reward, 1),
                  'Episode steps', episode_steps,
                  'Episode reward', round(episode_reward, 1))

            df = df.append(
                {
                    'Episode': len(reward_buffer),
                    'Reward': episode_reward,
                    'Steps': episode_steps,
                    'AvgReward': avg_reward,
                },
                ignore_index=True)

            episode_reward = 0
            episode_steps = 0
            env.reset()

        if len(reward_buffer) == max_episodes:
            df.to_csv(f'testing_sac.csv')
            print('Done!', np.round(np.mean(reward_buffer), 1))
            break


if __name__ == '__main__':
    # main_dqn_dueling_dqn()
    # testing_all_models(layers=4)
    testing_sac()
    # main_sac()
    pass
