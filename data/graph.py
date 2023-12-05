import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 导入 seaborn 库，用于更好的配色

# 设置 seaborn 风格
sns.set()

# 读取CSV文件
df = pd.read_csv('data\datasetWithLabel.csv')

# 按照epoch和label分组，统计每个组的数量
grouped_data = df.groupby(['epoch', 'label']).size().reset_index(name='count')

# 获取唯一的label和epoch值
labels = grouped_data['label'].unique()
epochs = grouped_data['epoch'].unique()

# 初始化柱状图和折线图的数据
bar_data = pd.DataFrame({'epoch': epochs, 'total_packets': 0})
line_data = grouped_data.pivot(index='epoch', columns='label', values='count').fillna(0)

# 计算每个epoch的总数据包数
bar_data['total_packets'] = grouped_data.groupby('epoch')['count'].sum().values

# 绘制图表
fig, ax1 = plt.subplots()
# plt.style.use('seaborn-whitegrid')
# fig.patch.set_facecolor('#f0f0f0')  # 浅灰色背景
# 柱状图表示每个epoch内数据包总数
ax1.bar(bar_data['epoch'], bar_data['total_packets'], color='b', alpha=0.7, label='Total Packets')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Total Packets', color='b')
ax1.tick_params('y', colors='b')

# 添加第二个纵坐标轴，用于绘制折线图
ax2 = ax1.twinx()

# 定义不同label对应的标记
markers = ['o', '^', 's', 'D', 'v', '*']

# 定义不同label对应的颜色
colors = sns.color_palette('husl', n_colors=len(labels))

# 绘制每种label对应的数据包个数折线图
for i, label in enumerate(labels):
    ax2.plot(line_data.index, line_data[label], label=f'{label} Packets', marker=markers[i], markersize=4, color=colors[i])

ax2.set_ylabel('Packet Count', color='g')
ax2.tick_params('y', colors='g')

# 设置图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# 显示图表
plt.show()
