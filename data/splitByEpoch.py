import csv
from collections import defaultdict

csv_file_path = 'datasetWithLabel.csv'  # 替换为你的CSV文件路径
output_folder = 'csv'  # 替换为你想保存切分文件的文件夹路径

with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)

    # 获取列索引
    header = next(csv_reader)
    epoch_index = header.index('epoch')

    # 创建一个字典来保存每个epoch的行
    epoch_rows = defaultdict(list)

    # 遍历每一行
    for row in csv_reader:
        epoch = row[epoch_index]
        epoch_rows[epoch].append(row)

# 根据epoch创建文件
for epoch, rows in epoch_rows.items():
    output_file_path = f'{output_folder}/epoch_{epoch}.csv'
    with open(output_file_path, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(header)
        csv_writer.writerows(rows)

print('CSV文件切分完成。')
