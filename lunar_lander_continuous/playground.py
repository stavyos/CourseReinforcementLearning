from collections import deque
from typing import Tuple

import cv2
import gym
import numpy as np
import pandas as pd

from architectures.sac.sac_agent import SacAgent

gym.logger.set_level(40)


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


def train_sac(agent: SacAgent, env: gym.Env, num_of_episodes: int, max_steps: int, start_episode: int = 1,
              avg_100: int = 0, test: bool = False):
    reward_buffer = deque(maxlen=100)

    if start_episode > 1:
        for i in range(101):
            reward_buffer.append(avg_100)

    if test:
        reward_buffer = deque(maxlen=100)

    cols = ['Episode', 'Reward', 'Steps', 'AvgReward', 'TotalSteps']
    df = pd.DataFrame(columns=cols)
    total_steps = 0
    for episode in range(start_episode, num_of_episodes + 1):
        reward, step, fail = run_one_episode_sac(agent=agent, env=env, max_steps=max_steps)

        if not test:
            cv2.imwrite(f'episodes\\{episode}.jpg', env.render('rgb_array'))
        reward_buffer.append(reward)
        total_steps += step

        print(
            f'Episode: {episode}, step: {step}, reward: {reward}, fail: {fail}, last 100 avg: {np.mean(reward_buffer)}')
        df = df.append(
            {
                'Episode': episode,
                'Reward': reward,
                'Steps': step,
                'AvgReward': np.mean(reward_buffer),
                'TotalSteps': total_steps
            },
            ignore_index=True)

        if episode % 20 == 0:
            if episode == start_episode:
                continue
            if not test:
                agent.save_model(data_dir='models', episode=episode, avg_100=int(np.mean(reward_buffer)))


def main():
    _env = gym.make('LunarLanderContinuous-v2')
    _agent = SacAgent(env=_env, hidden_size=256)

    train_sac(agent=_agent, env=_env, num_of_episodes=1000, max_steps=500)


if __name__ == '__main__':
    main()
