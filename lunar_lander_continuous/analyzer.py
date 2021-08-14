import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

legit_features = [25, 121, 225]


# noinspection PyUnusedLocal
def plotting_testing_avg_reward(_path: str, title: str):
    winner = -1, 0
    losser = -1, 1000

    output_features = []
    results = {}
    for network in os.listdir(_path):
        if not os.path.isdir(os.path.join(_path, network)):
            continue

        features = int(str.split(network, '_')[-1])

        if not legit_features.__contains__(features):
            continue

        output_features.append(features)

        df = pd.read_csv(filepath_or_buffer=os.path.join(_path, network, f'testing_{network}.csv'), sep=',')

        avg = df['AvgReward'].iloc[-1]
        if avg > winner[1]:
            winner = features, avg
        if avg < losser[1]:
            losser = features, avg

        results.update({features: df})

    output_features = sorted(output_features)

    data_feature = 'AvgReward'
    for features in output_features:

        winner_title, loser_title = '', ''

        if features == winner[0]:
            winner_title = ' $\\bf(Winner!)$'
        if features == losser[0]:
            loser_title = ' (Loser!)'

        extra_title = f'{winner_title}{loser_title}'

        df: pd.DataFrame = results.get(features)
        df = df[df['Episode'] >= 100]
        sns.lineplot(data=df[['Episode', data_feature]],
                     x='Episode',
                     y=data_feature, label=f'Quantization - {features}{extra_title}')

    plt.axhline(y=200,
                color='gray',
                linestyle='--',
                linewidth=1,
                alpha=0.75,
                label=f'Winning Threshold')

    plt.xlim((100, 1000))
    plt.ylim((0, 300))
    # plt.title(title, color='salmon', size=22, weight='bold')
    plt.xlabel('Episode', size=15)
    plt.ylabel('Avg Reward', size=15)
    # plt.legend()
    plt.legend()
    plt.legend(loc=4)
    plt.show()


def plotting_testing_avg_reward_dqn():
    for layers in [2, 3, 4]:
        _path = f'models//DQN//layers_{layers}//'
        title = f'DQN - Testing Models with {layers} Layers'
        plotting_testing_avg_reward(_path=_path, title=title)


def plotting_testing_avg_reward_dueling_dqn():
    _path = f'models//DuelingDQN//'
    title = f'Dueling DQN - Testing'
    plotting_testing_avg_reward(_path=_path, title=title)


# noinspection PyUnusedLocal
def plotting_avg_reward(_path: str, title: str):
    winner = -1, 1001
    losser = -1, -1, 1000
    min_episodes = 1000

    output_features = []
    results = {}
    for network in os.listdir(_path):
        if not os.path.isdir(os.path.join(_path, network)):
            continue

        features = int(str.split(network, '_')[-1])

        if not legit_features.__contains__(features):
            continue

        output_features.append(features)

        df = pd.read_csv(filepath_or_buffer=os.path.join(_path, network, 'logging.csv'), sep=',')

        min_episodes = min(min_episodes, df.shape[0])

        if winner[1] > df.shape[0]:
            winner = features, df.shape[0]

        if losser[1] < df.shape[0]:
            avg = df['AvgReward'].iloc[-1]
            losser = features, df.shape[0], avg
        elif losser[1] == df.shape[0]:
            avg = df['AvgReward'].iloc[-1]
            if losser[2] > avg:
                losser = features, df.shape[0], avg

        results.update({features: df})

    output_features = sorted(output_features)

    data_feature = 'AvgReward'
    for features in output_features:
        winner_title, loser_title = '', ''

        if features == winner[0]:
            winner_title = ' $\\bf(Winner!)$'
        if features == losser[0]:
            loser_title = ' (Loser!)'

        extra_title = f'{winner_title}{loser_title}'

        df: pd.DataFrame = results.get(features)
        print(features)
        print(df.iloc[-1])
        print('---------------------------------')
        sns.lineplot(data=df[['Episode', data_feature]],
                     x='Episode',
                     y=data_feature, label=f'Quantization - {features}{extra_title}')

    plt.axvline(x=min_episodes,
                color='gray',
                linestyle='--',
                linewidth=1,
                alpha=0.75,
                label=f'Episodes to WIN ({min_episodes})')
    # plt.title(title, color='salmon', size=22, weight='bold')
    plt.xlabel('Episode', size=15)
    plt.ylabel('Avg Reward', size=15)
    plt.legend()
    plt.show()


def plotting_avg_reward_dqn():
    for layers in [2, 3, 4]:
        _path = f'models//DQN//layers_{layers}//'
        title = f'DQN - Models with {layers} Layers'
        plotting_avg_reward(_path=_path, title=title)


def plotting_avg_reward_dueling_dqn():
    _path = f'models//DuelingDQN//'
    title = f'Dueling DQN'
    plotting_avg_reward(_path=_path, title=title)


def plotting_video_figure():
    df = pd.read_csv('models//SAC//logging.csv', sep=',')

    rewards = df['Reward'].to_numpy().astype('float')
    avg_rewards = df['AvgReward'].to_numpy().astype('float')
    solved_idx = np.where(avg_rewards >= 200)[0][0]

    # plt.plot(steps, label='Episode Steps')
    plt.plot(rewards, label='Epsiode Reward')
    plt.plot(avg_rewards, label='Avg 100 Last Rewards')
    plt.plot([200] * df.shape[0], label='Winning Game Threshold')
    plt.vlines(solved_idx,
               color='gray',
               ymin=-1000,
               ymax=2000,
               label=f'Enviroment Solved! ({solved_idx})',
               linestyle='--',
               linewidth=1,
               alpha=0.75)

    plt.xlim((0, 400))
    plt.ylim((rewards.min() - 50, 1050))
    plt.title('Solving LunarLanderContinuous-v2 using Soft Actor-Critic', color='salmon', size='22', weight='bold')
    plt.xlabel("Episode", size=16)
    plt.ylabel("Score", size=16)

    plt.legend()
    plt.show()


def main():
    # plotting_avg_reward_dqn()
    # plotting_avg_reward_dueling_dqn()
    # #
    plotting_testing_avg_reward_dqn()
    plotting_testing_avg_reward_dueling_dqn()
    # #
    # plotting_video_figure()
    pass


if __name__ == '__main__':
    main()
