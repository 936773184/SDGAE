import numpy as np
import copy
import torch
'''
def sparse_matrix(similarity_matrix,p):
    length=similarity_matrix.shape[0]#一定是方阵
    N=np.zeros((length,length))
    similarity_matrix_after_sparse=np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            pNeighborsofj=pNeighbors(j,similarity_matrix,p)
            pNeighborsofi=pNeighbors(i,similarity_matrix,p)

            if ( i in pNeighborsofj ) and ( j in pNeighborsofi ):
                N[i][j]=1
            elif (j not in pNeighborsofi ) and ( i not in pNeighborsofj ):
                N[i][j]=0
            else:
                N[i][j]=0.5
    # for i in range(length):
    #     for j in range(length):
    #         similarity_matrix_after_sparse[i][j]=similarity_matrix[i][j]*N[i][j]
    similarity_matrix_after_sparse=np.multiply(similarity_matrix,N)
    similarity_matrix_after_sparse=similarity_matrix_after_sparse+np.eye(length)
    return similarity_matrix_after_sparse


'''

def sparse_matrix(similarity_matrix,p):
    length=similarity_matrix.shape[0]#一定是方阵
    N=np.zeros((length,length))
    similarity_matrix_after_sparse=np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            pNeighborsofj=pNeighbors(j,similarity_matrix,p)
            pNeighborsofi=pNeighbors(i,similarity_matrix,p)

            if ( i in pNeighborsofj ) and ( j in pNeighborsofi ):
                N[i][j]=1
            elif (j not in pNeighborsofi ) and ( i not in pNeighborsofj ):
                N[i][j]=0
            else:
                N[i][j]=0.5
    # for i in range(length):
    #     for j in range(length):
    #         similarity_matrix_after_sparse[i][j]=similarity_matrix[i][j]*N[i][j]
    similarity_matrix_after_sparse=np.multiply(similarity_matrix,N)#这个地方有bug我靠靠靠靠靠靠！！！！！
    similarity_matrix_after_sparse=similarity_matrix_after_sparse+np.eye(length)
    return similarity_matrix_after_sparse



def pNeighbors(node,matrix,K):#根据相似性矩阵返回K近邻
    KknownNeighbors=np.array([])
    featureSimilarity=copy.deepcopy(matrix[node])#在相似性矩阵中取出第node行
    featureSimilarity[node]=-100   #排除自身结点,使相似度为-100
    KknownNeighbors=featureSimilarity.argsort()[::-1]#按照相似度降序排序
    KknownNeighbors=KknownNeighbors[:K]#返回前K个结点的下标
    return KknownNeighbors




if __name__ == "__main__":
    # Sd=np.loadtxt("whole_data/SD_708_708.txt")
    # St=np.loadtxt("whole_data/ST_1512_1512.txt")

    # Stemp=np.array([[1,0.3,0.6,0.1,0.9],
    #                 [0.3,1,0.4,0.5,0.9],
    #                 [0.6,0.4,1,0.8,0.1],
    #                 [0.1,0.5,0.8,1,0.3],
    #                 [0.9,0.9,0.1,0.3,1]
    #                 ])
    # sparse_matrix(similarity_matrix=Stemp,p=2)


    Stemp=np.array([[1,0.3,0.9,0.9,0.9],
                    [0.3,1,0.2,0.1,0.1],
                    [0.9,0.2,1,0.8,0.9],
                    [0.9,0.1,0.8,1,0.5],
                    [0.9,0.1,0.9,0.5,1]
                    ])
    sparse_matrix(similarity_matrix=Stemp,p=3)

    print("end")
