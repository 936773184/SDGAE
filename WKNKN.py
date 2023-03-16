import numpy as np
import math
import copy
def WKNKN(DTI,drugSimilarity,proteinSimilarity,K,r):
    drugCount=DTI.shape[0]
    proteinCount=DTI.shape[1]
    # 标志drug是new drug还是known drug 如果是known drug,则相应位为1
    # 如果是new drug,则相应位为0
    flagDrug=np.zeros([drugCount])
    flagProtein=np.zeros([proteinCount])
    for i in range(drugCount):
        for j in range(proteinCount):
            if(DTI[i][j]==1):
                flagDrug[i]=1
                flagProtein[j]=1
    Yd=np.zeros([drugCount,proteinCount])
    Yt=np.zeros([drugCount,proteinCount])
    # Yd矩阵的获取
    for d in range(drugCount):
        dnn=KNearestKnownNeighbors(d,drugSimilarity,K,flagDrug)#返回K近邻
        w=np.zeros([K])
        Zd=0
        # 获取权重w和归一化因子Zd
        for i in range(K):
            w[i]=math.pow(r,i)*drugSimilarity[d,dnn[i]]
            Zd+=drugSimilarity[d,dnn[i]]
        for i in range(K):
            Yd[d]=Yd[d]+w[i]*DTI[dnn[i]]
        Yd[d]=Yd[d]/Zd

    # Yt矩阵的获取
    for t in range(proteinCount):
        tnn=KNearestKnownNeighbors(t,proteinSimilarity,K,flagProtein)
        w=np.zeros([K])
        Zt=0
        for j in range(K):
            w[j]=math.pow(r,j)*proteinSimilarity[t,tnn[j]]
            Zt+=proteinSimilarity[t,tnn[j]]
        for j in range(K):
            Yt[:,t]=Yt[:,t]+w[j]*DTI[:,tnn[j]]
        Yt[:,t]=Yt[:,t]/Zt

    Ydt=Yd+Yt
    Ydt=Ydt/2

    ans=np.maximum(DTI,Ydt)#ans的形状是[drugCount,proteinCount]
    return ans
# 返回下标，node结点的K近邻（不包括new drug/new target）
def KNearestKnownNeighbors(node,matrix,K,flagNodeArray):
    KknownNeighbors=np.array([])
    featureSimilarity=matrix[node].copy()#在相似性矩阵中取出第node行
    featureSimilarity[node]=-100   #排除自身结点,使相似度为-100
    featureSimilarity[flagNodeArray==0]=-100  #排除new drug/new target,使其相似度为-100
    # 只考虑known node
    KknownNeighbors=featureSimilarity.argsort()[::-1]#按照相似度降序排序
    KknownNeighbors=KknownNeighbors[:K]#返回前K个结点的下标
    return KknownNeighbors
if __name__ == "__main__":
    # DTI=np.array([[1,1,0],[1,0,0],[0,0,1],[0,0,0]],dtype=float)
    # Sd=np.array([[1,0.7,0.8,0],[0.7,1,0.6,0.6],[0.8,0.6,1,0.5],[0,0.6,0.5,1]],dtype=float)
    # St=np.array([[1,0.5,0.4],[0.5,1,0],[0.4,0,1]],dtype=float)
    # predict_Y=WKNKN(DTI,Sd,St,K=2,r=0.7)
    DTI=np.loadtxt('whole_data/DTI_708_1512.txt')
    # Sd=np.loadtxt('whole_data/SD_708_708.txt')
    # St=np.loadtxt('whole_data/ST_1512_1512.txt')
    Sd=np.loadtxt('whole_data/multi_similarity/drug_fusion_similarity_708_708.txt')
    St=np.loadtxt('whole_data/multi_similarity/target_fusion_similarity_1512_1512.txt')
    predict_Y=WKNKN(DTI=DTI,drugSimilarity=Sd,proteinSimilarity=St,K=10,r=0.8)
    # 统计原始数据集非0的个数
    num1=0
    for i in range(DTI.shape[0]):
        for j in range(DTI.shape[1]):
            if DTI[i][j]==1:
                num1+=1
    frequent_no_zero=num1/(DTI.shape[0]*DTI.shape[1])
    print("Original data none zero ratio:%.4f"%frequent_no_zero)
    num_float=0
    for i in range(predict_Y.shape[0]):
        for j in range(predict_Y.shape[1]):
            if predict_Y[i][j]!=0:
                num_float+=1
    frequent_no_zero=num_float/(predict_Y.shape[0]*predict_Y.shape[1])
    print("After WKNKN,none zero ratio:%.4f"%frequent_no_zero)
    # 离散化WKNKN


    num_greaterzero_samllerone=0
    for i in range(predict_Y.shape[0]):
        for j in range(predict_Y.shape[1]):
            if predict_Y[i][j]!=0 and predict_Y[i][j]!=1:
                num_greaterzero_samllerone += 1

    print('Before discretize: num of float num %d'%num_greaterzero_samllerone)

    float_array=copy.deepcopy(predict_Y[ (predict_Y>0) & (predict_Y<1) ])
    float_median=np.median(float_array)

    predict_Y[predict_Y>=float_median]=1
    predict_Y[predict_Y<float_median]=0

    num_greaterzero_samllerone=0
    for i in range(predict_Y.shape[0]):
        for j in range(predict_Y.shape[1]):
            if predict_Y[i][j]!=0 and predict_Y[i][j]!=1:
                num_greaterzero_samllerone += 1
    print('After discretize: num of float %d'%num_greaterzero_samllerone)




    np.savetxt('whole_data/multi_similarity/DTI_708_1512_WKNKN_MAX_DISCRETIZE.txt',predict_Y)


    print('end!!')




