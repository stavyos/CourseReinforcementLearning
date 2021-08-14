import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

sns.set_theme()

#
# steps = df[3].astype(dtype='int').to_numpy()
# rewards = df[5].astype(dtype='float').to_numpy()
# avg_100 = df[11].astype(dtype='float').to_numpy()
#
# new_df = pd.DataFrame(data={'Episode': np.arange(1, steps.shape[0] + 1).astype('float'),
#                             'Reward': rewards,
#                             'Steps': steps,
#                             'AvgReward': avg_100,
#                             'TotalSteps': np.cumsum(steps)
#                             })

# new_df.to_csv('logging.csv', sep=',', index=False)

df = pd.read_csv('logging.csv', sep=',')


episodes = df['Episode'].to_numpy().astype('int')
rewards = df['Reward'].to_numpy().astype('float')
steps = df['Steps'].to_numpy().astype('int')
avg_rewards = df['AvgReward'].to_numpy().astype('float')
total_steps = df['TotalSteps'].to_numpy().astype('int')
solved_idx = np.where(avg_rewards >= 200)[0][0]

plt.plot(steps, label='Episode Steps')
plt.plot(rewards, label='Epsiode Reward')
plt.plot(avg_rewards, label='Avg 100 Last Rewards')
plt.plot([200] * df.shape[0], label='Winning Game Threshold')
plt.vlines(solved_idx,
           colors='black',
           ymin=rewards.min(),
           ymax=1000,
           label=f'Enviroment Solved! ({solved_idx})',
           linestyles='--')

plt.xlim((0, 400))
plt.ylim((rewards.min() - 50, 1050))
plt.title('LunarLanderContinuous-v2', color='salmon', size='22', weight='bold')
plt.xlabel("Episode", size=16)
plt.ylabel("Score", size=16)

plt.legend()
plt.show()
