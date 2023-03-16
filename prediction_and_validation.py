from my_function import *
import numpy as np
from pylab import *
from sklearn.metrics import auc, roc_curve, precision_recall_curve,roc_auc_score,average_precision_score
from tqdm import tqdm
# FOLD = 5
from matplotlib import pyplot as plt
import torch
def read_dict(path):
    dicFile = open(path, 'r')  # 打开数据
    txtDict = {}  # 建立字典
    while True:
        line = dicFile.readline().strip('\n')#去除前后的换行符
        if line == '':
            break
        index = line.find(':')  # 以tab键为分割
        key = line[:index]
        value = line[index+1:]
        # if key in txtDict:
        #     txtDict[key]+=1
        # else:
        #     txtDict[key]=1
        txtDict[key] = value  # 加入字典
    dicFile.close()
    return txtDict

DTI = np.loadtxt("whole_data/DTI_708_1512.txt")#标签矩阵
index1 = np.loadtxt('divide_result/index_1.txt')
index0 = np.loadtxt('divide_result/index_0.txt')
index = np.hstack((index1, index0))
drug_num = DTI.shape[0]#708个药物
protein_num = DTI.shape[1]#1512个靶点
score = np.zeros(DTI.shape)
for f in tqdm(range(5)):
    pre = np.loadtxt('predict_result_of_ARGA/ARGA'+str(f)+'.txt')
    idx = index[f, :]
    # pre = line_normalize(pre)
    # pre = pre[:, 1]
    for i in range(len(idx)):
        d = int(idx[i] / protein_num)
        p = int(idx[i] % protein_num)
        score[d, p] += pre[i]
# 对score排序
TOPK=30
drug_id=np.loadtxt('whole_data/drug_708.txt',dtype=str)
protein_id=np.loadtxt('whole_data/protein_1512.txt',dtype=str)
score_tensor=torch.tensor(score)
# 预测分数 预测靶点下标
# candidate_target_score.shape=708,30
# candidate_target_index.shape=708,30
candidate_target_score,candidate_target_index=score_tensor.topk(k=TOPK,dim=1)
candidate_target_score=np.array(candidate_target_score)
candidate_target_index=np.array(candidate_target_index)

ans=np.empty(shape=(0,6))
drug_dict_path='whole_data/other_information_to_be_used/drug_dict_map.txt'
target_dict_path='whole_data/other_information_to_be_used/protein_dict_map.txt'
drug_dict=read_dict(drug_dict_path)
target_dict=read_dict(target_dict_path)









for i in range(drug_num):
    drug_id_drugbank=drug_id[i]
    first_column=np.array( [drug_id_drugbank]*30 )
    second_column=np.arange(start=1,stop=TOPK+1,step=1)
    third_column=protein_id[candidate_target_index[i]]
    forth_column=candidate_target_score[i]
    fifth_column=np.array([drug_dict[drug_id_drugbank]]*30)
    sixth_column=np.array(list(map(target_dict.get,third_column)))#根据字典映射成蛋白名
    # 扩展维度，方便拼接
    first_column=np.expand_dims(first_column,axis=1)
    second_column=np.expand_dims(second_column,axis=1)
    third_column=np.expand_dims(third_column,axis=1)
    forth_column=np.expand_dims(forth_column,axis=1)
    fifth_column=np.expand_dims(fifth_column,axis=1)
    sixth_column=np.expand_dims(sixth_column,axis=1)
    drug_candidate_target=np.concatenate((first_column,second_column,third_column,forth_column,fifth_column,sixth_column),axis=1)
    ans=np.concatenate((ans,drug_candidate_target),axis=0)

np.savetxt("predict_candidate_target_include_name.csv",ans,fmt="%s",delimiter=",")
# np.savetxt()



print('end')












'''
DTI=DTI.tolist()
score=score.tolist()
auc_list = []
aupr_list = []
tpr_list = []
fpr_list = []
recall_list = []
precision_list = []
c=0
for i in tqdm(range(drug_num)):#针对每一个药物而言
    if np.sum(np.array(DTI[i])) == 0:
        c += 1
        continue
    else:
        tpr1, fpr1, precision1, recall1 = tpr_fpr_precision_recall(np.array(DTI[i]), np.array(score[i]))
        fpr_list.append(fpr1)
        tpr_list.append(tpr1)
        precision_list.append(precision1)
        recall_list.append(recall1)
        auc_list.append(auc(fpr1, tpr1))
        aupr_list.append(auc(recall1, precision1)+recall1[0]*precision1[0])

tpr = equal_len_list(tpr_list)
fpr = equal_len_list(fpr_list)
precision = equal_len_list(precision_list)
recall = equal_len_list(recall_list)
tpr=np.array(tpr)
fpr=np.array(fpr)
precision=np.array(precision)
recall=np.array(recall)


tpr_mean = np.mean(tpr, axis=0)
fpr_mean = np.mean(fpr, axis=0)
recall_mean = np.mean(recall, axis=0)
precision_mean = np.mean(precision, axis=0)
AUC=auc(fpr_mean, tpr_mean)
AUPR=auc(recall_mean, precision_mean)+recall_mean[0]*precision_mean[0] #第(recall_mean[0],precision_mean[0])点的P值最高，R值最低，也就是PR曲线最左端的点
print('The auc of prediction is:%.4f'%AUC)
print('The aupr of prediction is:%.4f'%AUPR)
# 画ROC曲线

plt.figure()
plt.plot(fpr_mean,tpr_mean,label='ROC(AUC = %0.4f)' % AUC)
plt.title('ROC curve')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc="lower right")
plt.savefig('final_result_evaluation/ROC.jpg')
plt.show()

# 画PR曲线

plt.figure()
plt.plot(recall_mean,precision_mean,label='PR(AUPR = %0.4f)' % AUPR)
plt.title('PR curve')
plt.xlabel("RECALL")
plt.ylabel("PRECISION")
plt.legend(loc="lower right")
plt.savefig('final_result_evaluation/PR.jpg')
plt.show()

with open('final_result_evaluation/metrics.txt', 'w') as f:
    print('AUC:%.6f '%AUC,  'AUPR: %.6f'%AUPR,  file=f)

print('end')
'''



'''

y_true = DTI.flatten()
y_score = score.flatten()

# 预测AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
AUC_ROC = roc_auc_score(y_true, y_score)
plt.figure()
plt.plot(fpr,tpr,label='ROC(AUC = %0.4f)' % AUC_ROC)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC curve')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc="lower right")
plt.savefig('final_result/ROC.jpg')
plt.show()
# with open('final_result/metric.txt', 'a') as f:
print('AUC:%.6f'%AUC_ROC)

# 预测AUPR---------10.19写这个

precision, recall, threshold = precision_recall_curve(y_true, y_score, pos_label=1)
AUPR=average_precision_score(y_true,y_score)
# precision=precision[:-2]
# recall=recall[:-2]
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
plt.figure()
plt.plot(recall,precision,label='PR(AUPR = %0.4f)'% AUPR)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('PR curve')
plt.xlabel("RECALL")
plt.ylabel("PRECISION")
plt.legend(loc="lower right")
plt.savefig('final_result/PR.jpg')
plt.show()
print('AUPR:%.6f'%AUPR)






'''
















