import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.sparse as sp
from GR import sparse_matrix,pNeighbors
import numpy as np
from my_function import *
from torch_geometric.nn import GCNConv
import copy
import sys
import argparse
import os
#@@@@@@@@@@@@@@@@@@@@@
parser=argparse.ArgumentParser()
parser.add_argument('-f',dest="fold",type=int)
results=parser.parse_args()
fold =results.fold #脚本的第一个参数指示当前运行的是第几折
#@@@@@@@@@@@@@@@@@@@@@@@@@@

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
EPOCH = 5000 #训练epoch数
BATCH_SIZE = 64           # 48*64
LR = 0.0001               # 药物0.01，靶标0.001
# LR = 0.05               #学习率  药物0.01，靶标0.001
# n = 549
# m = 424
n = 708 #708个药物
m = 1512 #1512个靶点

in_features = 2220
out_features = 200
# 图卷积第一层结束时特征数
N_HID = 500

adjust_p_neighbors_parameters=False


lambda_l=0.00001
lambda_d=0.001
lambda_t=0.001
# 对称归一化的laplacian矩阵,输入是邻接矩阵，输出是对称归一化后的拉普拉斯矩阵
def normalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1)
    R_sqrt = 1 / np.sqrt(R)
    D_sqrt = np.diag(R_sqrt)
    I = np.eye(adj_matrix.shape[0])
    return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)



# A = np.loadtxt('data/DTI.txt')
# SR = np.loadtxt('data/SD.txt')
# SP = np.loadtxt('data/ST.txt')
# X0 = np.hstack((SR, A))
# X1 = np.hstack((A.T, SP))
# X = np.vstack((X0, X1))
# TTI = np.loadtxt('data/TTI.txt')
# DDI= np.loadtxt('data/DDI.txt')
# A0 = np.hstack((DDI, A))
# A1 = np.hstack((A.T, TTI))
# A = np.vstack((A0, A1))


A = np.loadtxt('divide_result/A'+str(fold)+'.txt')#其中A有WKNKN的信息
X = np.loadtxt('divide_result/X'+str(fold)+'.txt')#其中X有融合相似度的信息

edge_index_temp=sp.coo_matrix(A)
edge_weight= copy.deepcopy(edge_index_temp.data)  #边权
edge_weight=torch.FloatTensor(edge_weight).to(device)#将numpy转为tensor 我们要利用的边权
edge_index_A= np.vstack((edge_index_temp.row, edge_index_temp.col))#提取的边[2,num_edges]
edge_index_A=torch.LongTensor(edge_index_A).to(device) #将numpy转为tensor 我们要利用的边的index







if adjust_p_neighbors_parameters==True:
    Sd = np.loadtxt('whole_data/multi_similarity/drug_fusion_similarity_708_708.txt')
    St = np.loadtxt('whole_data/multi_similarity/target_fusion_similarity_1512_1512.txt')
    Sd_after_sparse = sparse_matrix(similarity_matrix=Sd, p=5)
    St_after_sparse = sparse_matrix(similarity_matrix=St, p=5)
    # np.savetxt('whole_data/SD_708_708_after_sparse.txt', Sd_after_sparse)
    # np.savetxt('whole_data/ST_1512_1512_after_sparse.txt', St_after_sparse)
    np.savetxt('whole_data/multi_similarity/drug_fusion_similarity_max_708_708_after_sparse.txt',Sd_after_sparse)
    np.savetxt('whole_data/multi_similarity/target_fusion_similarity_max_1512_1512_after_sparse.txt',St_after_sparse)
else:
    Sd_after_sparse=np.loadtxt('whole_data/multi_similarity/drug_fusion_similarity_max_708_708_after_sparse.txt')
    St_after_sparse=np.loadtxt('whole_data/multi_similarity/target_fusion_similarity_max_1512_1512_after_sparse.txt')

Ld=normalized_laplacian(Sd_after_sparse)
Lt=normalized_laplacian(St_after_sparse)







# 构建拉普拉斯矩阵 A = D（A+I）D
A_laplacians = laplacians(A)#此时的A矩阵已经包含了WNKN的消息


# 图卷积操作
class GraphConvolution(nn.Module):
    def __init__(self, in_size, out_size,):
        super(GraphConvolution, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.FloatTensor(in_size, out_size))#图卷积的W矩阵
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, X, A):#X是结点特征矩阵 A是邻接矩阵（调用的时候是结点特征矩阵在前，邻接矩阵在后）
        support = torch.mm(X, self.weight)  # X*W
        result = torch.spmm(A, support)    # A*X*W
        return result


# 两个图卷积类构成的双层图卷积
'''
class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_class):     # 原始特征数 第一层结束特征数 最终特征数
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(n_feat, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_class)

        # self.dropout = dropout

    def forward(self, x, A):
        x = self.gc1(x, A)#调用的时候是结点特征矩阵在前，邻接矩阵在后
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.gelu(x)#原代码
        x = torch.sigmoid(x)
        # x=F.relu(x)  #此处论文的激活函数应该是relu
        x = self.gc2(x, A)
        # x = F.sigmoid(x)#原来的代码
        # x=torch.sigmoid(x)
        # x=torch.relu(x)#图卷积最后一层用relu
        x=torch.tanh(x)#图卷积最后一层用tanh
        return x

'''
class GCN_PYG(nn.Module):
    def __init__(self, n_feat, n_hid, n_class):     # 原始特征数 第一层结束特征数 最终特征数
        super(GCN_PYG, self).__init__()
        self.gc1 = GCNConv(in_channels=n_feat,out_channels=n_hid,add_self_loops=True) #邻接矩阵中没有自连，我们还是加上自连吧
        self.gc2 = GCNConv(in_channels=n_hid,out_channels=n_class,add_self_loops=True) #

    def forward(self, x, edge_index, edge_weight): #x:结点特征矩阵 edge_index:coo格式存储的边索引 edge_weight：边权
        x = self.gc1(x,edge_index,edge_weight)
        x = torch.sigmoid(x)
        # x = torch.relu(x)
        x = self.gc2(x,edge_index,edge_weight)
        x = torch.tanh(x)
        return x









# Discriminator 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(out_features, 64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64, 2),#多分类最后一层不需要加激活函数，因为nn.CrossEntropyLoss可以自动softmax
            # nn.Sigmoid()
            # nn.ReLU()#二分类 由于使用nn.CrossEntropyLoss作为损失函数，所以最后一层不用softmax
            )

    def forward(self, x):
        x = self.dis(x)
        return x


# 解码器
class Decoder(nn.Module):
    def forward(self, x):
        y = x.permute(1, 0)
        z = torch.mm(x, y)
        z = torch.Tensor.relu(z)#此处论文中激活函数应该是sigmod
        # z = torch.sigmoid(z)
        return z


# Model and optimizer
decoder = Decoder()
D = Discriminator()
G = GCN_PYG(n_feat=in_features,
        n_hid=N_HID,
        n_class=out_features,
        # dropout=DROPOUT
        )

loss_function_E = nn.MSELoss()#
loss_function_G = nn.CrossEntropyLoss()
loss_function_D = nn.CrossEntropyLoss()

D_optimizer = torch.optim.Adam(D.parameters(), lr=LR*0.1)
G_optimizer = torch.optim.Adam(G.parameters(), lr=LR)

A_laplacians = torch.from_numpy(A_laplacians).float().view(n+m, n+m)
X = torch.from_numpy(X).float().view(n+m, n+m)#此时的X矩阵已经有多源相似度融合的信息

# if torch.cuda.is_available():
#     # 尝试将模型放入GPU中
#     decoder = decoder.cuda()
#     G = G.cuda()
#     D = D.cuda()
#     # 尝试将数据放入GPU中
#     A_laplacians = A_laplacians.cuda()
#     X = X.cuda()

decoder=decoder.to(device)
G=G.to(device)
D=D.to(device)
A_laplacians=A_laplacians.to(device)
X=X.to(device)









def loss_GRMF(Z,Ld,Lt,drug_num,lambda_l,lambda_d,lambda_t):
    A=Z[:drug_num]#前drug_num行代表药物的特征 A矩阵代表药物的特征
    B=Z[drug_num:]#后面代表靶点的特征 B矩阵代表靶点的特征
    A_loss = torch.norm(A)
    B_loss= torch.norm(B)
    A_loss=torch.square(A_loss)
    B_loss=torch.square(B_loss)
    loss_L2= A_loss+B_loss
    # Ld=normalized_laplacian(Sd_after_sparse) #用CPU计算拉普拉斯矩阵太慢了，尝试一下更改成GPU
    # Lt=normalized_laplacian(St_after_sparse)
    Ld=torch.tensor(np.float32(Ld)).to(device) #将numpy转成tensor
    Lt=torch.tensor(np.float32(Lt)).to(device)

    loss_drug= torch.mm(torch.mm(A.t(),Ld),A).trace()
    loss_target=  torch.mm(torch.mm(B.t(),Lt),B).trace()

    return lambda_l*loss_L2+lambda_d*loss_drug+lambda_t*loss_target




















# 开始训练
for epoch in range(EPOCH):
    # Z = G(A_laplacians, X)                 # 通过G得到（A，X）的编码结果Z  2220*200
    # Z = G(X , A_laplacians)
    Z = G(X,edge_index_A,edge_weight)
    A_hat = decoder(Z)     #解码
    # 在这里增加GRMF的损失项，在这里增加损失项主要是为了规范化
    G_loss1=loss_function_E(A_hat, torch.from_numpy(A).float().view(n + m, n + m).to(device))
    G_loss2=loss_GRMF(Z=Z,Ld=Ld,Lt=Lt,drug_num=n,lambda_l=lambda_l,lambda_d=lambda_d,lambda_t=lambda_t)
    G_loss=G_loss1+G_loss2

    # G_loss = loss_function_E(A_hat, torch.from_numpy(A).float().view(n + m, n + m).to(device))  \
    #          + loss_GRMF(Z=Z,Ld=Ld,Lt=Lt,drug_num=n,lambda_l=lambda_l,lambda_d=lambda_d,lambda_t=lambda_t)  # 计算重构误差     # 此时的A矩阵是有小数的

    # G_loss = loss_function_E(A_hat, torch.from_numpy(A).float().view(n+m, n+m).cuda())       # 计算重构误差
    G_optimizer.zero_grad()
    G_loss.backward()
    G_optimizer.step()
    # with open('log.txt', 'a') as f:
    #     print('Epoch: ', epoch, '| train G_loss: %.10f' % G_loss.item(),file=f)
    print('Epoch: ', epoch, '| train Encoding_loss: %.10f' % G_loss.item(),"G_loss1: %.10f" % G_loss1.item(),"G_loss2: %.10f" % G_loss2.item() ,end=' ')



    Z = Z.data.cpu()  # 2220*200
    # =================train discriminator 判别器
    # GPU版本
    # real_data = Variable(torch.randn(n+m, out_features)).cuda()     # 随机生成973个200维，服从正太分布的向量
    # fake_data = Variable(Z).cuda()
    # real_label = Variable(torch.ones(n + m)).cuda()                # 为输入数据生成标签（真1假0）
    # fake_label = Variable(torch.zeros(n + m)).cuda()

    # CPU版本
    real_data = Variable(torch.randn(n + m, out_features)).to(device)  # 随机生成2220个200维，服从正太分布的向量
    fake_data = Variable(Z).to(device)
    real_label = Variable(torch.ones(n + m)).to(device)  # 为输入数据生成标签（真1假0）
    fake_label = Variable(torch.zeros(n + m)).to(device)

    # compute loss of real_data
    real_out = D(real_data)
    d_loss_real = loss_function_D(real_out, torch.Tensor.long(real_label).to(device))
    real_scores = real_out  # closer to 1 means better

    # compute loss of fake_data
    fake_out = D(fake_data)
    d_loss_fake = loss_function_D(fake_out, torch.Tensor.long(fake_label).to(device))
    fake_scores = fake_out  # closer to 0 means better

    # 更新判别器
    D_loss = d_loss_real + d_loss_fake
    print('| train Discriminator_loss: %.10f' % D_loss.item(), end=' ')
    D_optimizer.zero_grad()
    D_loss.backward()
    D_optimizer.step()

    # ===============train generator 生成器
    # compute loss of fake_img
    output = D(fake_data)
    G_loss = loss_function_G(output, torch.Tensor.long(real_label))
    print('| train Generator_loss: %.10f' % G_loss.item(),end='\n')

    # 更新生成器
    G_optimizer.zero_grad()
    G_loss.backward()
    G_optimizer.step()


result = A_hat.data.cpu().numpy()
embedding = Z.data.cpu().numpy()
np.savetxt('predict_result_of_ARGA/score'+str(fold)+'.txt', result)
np.savetxt('predict_result_of_ARGA/embedding'+str(fold)+'.txt', embedding)
