from scipy.spatial import distance
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import copy
# 计算相似度并使用最大最小归一化
def compute_distance(feature):
    node_num=feature.shape[0]
    similarity=np.zeros((node_num,node_num))
    for i in range(node_num):
        for j in range(i,node_num):
            euclidean_distance=distance.euclidean(feature[i],feature[j])
            node_node_similarity=1/(1+euclidean_distance)
            similarity[i][j]=node_node_similarity
    # 将上三角矩阵转为对称阵
    similarity += similarity.T - np.diag(similarity.diagonal())
    # 对相似度进行归一化 最小最大归一化
    for i in range(node_num):
        max_similarity=1
        min_similarity=np.min(similarity[i])#按行进行归一化
        similarity[i]=(similarity[i]-min_similarity)/(max_similarity-min_similarity)
    return similarity




drug_num=708
target_num=1512
select_drug_num=20#选取的药物的数量
drug_feature_withGRMF=np.loadtxt('embedding0_withGRMF.txt')
drug_feature_withoutGRMF=np.loadtxt('embedding0_withoutGRMF.txt')
drug_feature_withGRMF=drug_feature_withGRMF[:drug_num]
drug_feature_withoutGRMF=drug_feature_withoutGRMF[:drug_num]
multisimilarity=np.loadtxt('../../whole_data/multi_similarity/drug_fusion_similarity_708_708.txt')

seed=30
np.random.seed(seed)
drug_index=np.random.randint(low=0,high=drug_num,size=select_drug_num)#取不到drug_num

select_drug_feature_withGRMF = copy.deepcopy(drug_feature_withGRMF[drug_index]) #这两行的代码的主要作用是为了选取20个药物
select_drug_feature_withoutGRMF = copy.deepcopy(drug_feature_withoutGRMF[drug_index]) #这两行的代码的主要作用是为了选取20个药物

# select_drug_feature_withGRMF = copy.deepcopy(drug_feature_withGRMF)
# select_drug_feature_withoutGRMF = copy.deepcopy(drug_feature_withoutGRMF)


similarity_withGRMF=compute_distance(select_drug_feature_withGRMF)
similarity_withoutGRMF=compute_distance(select_drug_feature_withoutGRMF)

# 在原始空间中筛选出特定的药物
multisimilarity=multisimilarity[drug_index]#取出对应的行
multisimilarity=multisimilarity[:,drug_index]#取出对应的列

# 对原始空间也进行一下最大最小归一化
for i in range(select_drug_num):
    max_similarity = 1
    min_similarity = np.min(multisimilarity[i])  # 按行进行归一化
    multisimilarity[i] = (multisimilarity[i] - min_similarity) / (max_similarity - min_similarity)



plot=sns.heatmap(pd.DataFrame(multisimilarity),cmap='Reds')
plt.savefig('original_space_similarity_20_drug_seed'+str(seed)+'.jpg')
plt.show()

plot=sns.heatmap(pd.DataFrame(similarity_withGRMF),cmap='Reds')
plt.savefig('withGRMF_similarity_20_drug_seed'+str(seed)+'.jpg')
plt.show()

plot=sns.heatmap(pd.DataFrame(similarity_withoutGRMF),cmap='Reds')
plt.savefig('withoutGRMF_similarity_20_drug_seed'+str(seed)+'.jpg')
plt.show()
print('end')













