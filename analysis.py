import pandas as pd

# 读取CSV文件（假设文件名为data.csv）
df = pd.read_csv('csv\epoch_25.csv')

# 统计label列的不同值的个数
label_counts = df['label'].value_counts(dropna=False)

# 打印每个值及其个数
for label, count in label_counts.items():
    print(f'Label: {label}, Count: {count}')

