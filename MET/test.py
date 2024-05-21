import torch
import torch.nn as nn

# 定义一个嵌入层
embedding = nn.Embedding(num_embeddings=10, embedding_dim=3)

# 定义一个包含索引的输入张量
input_indices = torch.tensor([1, 2, 3, 4])

# 获取嵌入向量
output = embedding(input_indices)

print(output)
