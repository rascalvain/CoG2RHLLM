import pandas as pd

# 读取 CSV 文件
file_path = './val.csv'  # 替换为你的 CSV 文件路径
df = pd.read_csv(file_path)

# 删除 'type' 列为 0 的行
df_filtered = df[df['type'] != 0]

# 将结果保存到新的 CSV 文件
output_file_path = './val_filter.csv'  # 替换为你想要保存的文件名
df_filtered.to_csv(output_file_path, index=False)

print(f"已删除 'type' 列为 0 的行，并保存到 {output_file_path}")
