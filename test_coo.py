import scipy.sparse as sp
import numpy as np
import torch


A = np.array([[0, 3, 0, 1],
              [1, 0, 2, 0],
              [0, 1, 0, 0],
              [1, 0, 0, 0]])

edge_index_temp = sp.coo_matrix(A)
print(edge_index_temp)

values = edge_index_temp.data  # 边上对应权重值weight
indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
edge_index_A = torch.LongTensor(indices)  # 我们真正需要的coo形式
print(edge_index_A)

i = torch.LongTensor(indices)  # 转tensor
v = torch.FloatTensor(values)  # 转tensor 边权



# edge_index = torch.sparse_coo_tensor(i, v, edge_index_temp.shape)
# print(edge_index)


