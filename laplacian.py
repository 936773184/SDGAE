
import numpy as np
import torch

def normalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1)
    R_sqrt = 1 / np.sqrt(R)
    D_sqrt = np.diag(R_sqrt)
    I = np.eye(adj_matrix.shape[0])
    return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)

# 此时adj_matrix应输入张量
def normalized_laplacian_gpu(adj_matrix):
    R=torch.sum(adj_matrix,dim=1)
    R_sqrt=1/torch.sqrt(R)
    D_sqrt=torch.diag_embed(R_sqrt)
    I=torch.eye(adj_matrix.shape[0])
    return I - torch.mm(torch.mm(D_sqrt,adj_matrix),D_sqrt)



    # R = np.sum(adj_matrix, axis=1)
    # R_sqrt = 1 / np.sqrt(R)
    # D_sqrt = np.diag(R_sqrt)
    # I = np.eye(adj_matrix.shape[0])
    # return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)
