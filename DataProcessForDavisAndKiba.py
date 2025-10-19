import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('davis_all.csv')

# 方法2：使用numpy的choice函数
split_categories = ['train', 'val', 'test']
probabilities = [0.7, 0.2, 0.1]

# 为每行随机分配split值
df['split'] = np.random.choice(split_categories, size=len(df), p=probabilities)

# 保存文件
df.to_csv('davis_all_with_split.csv', index=False)

# 验证结果
print("Split分布统计:")
print(df['split'].value_counts(normalize=True))
print("\n实际数量:")
print(df['split'].value_counts())