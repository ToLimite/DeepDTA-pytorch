import pandas as pd

# 读取CSV文件
df = pd.read_csv('kiba_all_with_split.csv')  # 替换为你的文件路径

# 计算最长的字符数
max_protein_length = df['proteins'].astype(str).str.len().max()
max_ligand_length = df['ligands'].astype(str).str.len().max()

print(f"proteins列中最长的字符数: {max_protein_length}")
print(f"ligands列中最长的字符数: {max_ligand_length}")