import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 文件路径
files = {
    'ACO_all': 'ACO_results_antnum350_all.csv',
    'ACO_nocar': 'ACO_results_antnum350_nocar.csv',
    'ACO_notallmodes': 'ACO_results_antnum350_notallmodes.csv',
    'DQN_all': 'DQN_experiment_results_episode400_all.csv',
    'DQN_nocar': 'DQN_experiment_results_episode400_nocar.csv',
    'DQN_notallmodes': 'DQN_experiment_results_episode400_notallmodes.csv',
    'Q_learning_all': 'Q_learning_results_episode400_all.csv',
    'Q_learning_nocar': 'Q_learning_results_episode400_nocar.csv',
    'Q_learning_notallmodes': 'Q_learning_results_episode400_notallmodes.csv'
}

# 加载数据并修改列名
data = {}
for key, filepath in files.items():
    df = pd.read_csv(filepath)
    if 'DQN' in key:
        df = df[df['Find']]  # DQN数据只选择Find为True的
    if 'Time Cost' in df.columns and not 'Travel Time Cost (seconds)' in df.columns:
        df.rename(columns={'Time Cost': 'Travel Time Cost (seconds)'}, inplace=True)
    data[key] = df

# 设置绘图样式
sns.set(style="whitegrid")

# 绘制图形，确保有3行来表示三种算法
fig, ax = plt.subplots(3, 3, figsize=(18, 18), sharex='col', sharey='row')
algorithms = ['ACO', 'DQN', 'Q_learning']
conditions = ['all', 'nocar', 'notallmodes']
condition_titles = ['all modes available', 'no e-car stations', 'not all modes available']

for i, algo in enumerate(algorithms):
    for j, cond in enumerate(conditions):
        ax_key = f'{algo}_{cond}'
        # sns.histplot(data[ax_key]['Travel Time Cost (seconds)'], ax=ax[i, j], color='blue', kde=True)
        sns.histplot(data[ax_key]['Execution Time (seconds)'], ax=ax[i, j], color='red', kde=True)
        ax[i, j].set_title(f'{algo} - {condition_titles[j]}')
        ax[i, j].set_xlabel('Execution Time (seconds)')  # 设置横轴标签
        ax[i, j].set_ylabel('Frequency')  # 设置纵轴标签

plt.tight_layout()
plt.show()