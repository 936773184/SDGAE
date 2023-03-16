from my_function import *
import numpy as np
from pylab import *
from sklearn.metrics import auc, roc_curve, precision_recall_curve,roc_auc_score,average_precision_score
from tqdm import tqdm
import torch
import copy
# FOLD = 5
from matplotlib import pyplot as plt


DTI = np.loadtxt("../../whole_data/DTI_708_1512.txt")#ground truth标签矩阵
index1 = np.loadtxt('../../divide_result/index_1.txt')
index0 = np.loadtxt('../../divide_result/index_0.txt')
index = np.hstack((index1, index0))
drug_num = DTI.shape[0]#708个药物
protein_num = DTI.shape[1]#1512个靶点
score = np.zeros(DTI.shape)
for f in tqdm(range(5)):
    pre = np.loadtxt('../../predict_result_of_ARGA/ARGA'+str(f)+'.txt')
    idx = index[f, :]
    # pre = line_normalize(pre)
    # pre = pre[:, 1]
    for i in range(len(idx)):
        d = int(idx[i] / protein_num)
        p = int(idx[i] % protein_num)
        score[d, p] += pre[i]

'''
DTI=DTI.tolist()
score=score.tolist()
auc_list = []
aupr_list = []
tpr_list = []
fpr_list = []
recall_list = []
precision_list = []


'''

TOPK_precent=0.30
TOPK=int( protein_num*TOPK_precent )#取TOPK个候选靶点

all_drugs_recall=[]


score_tensor=torch.tensor(score)
# 预测分数 预测靶点下标
# candidate_target_score.shape=708,30
# candidate_target_index.shape=708,30
candidate_target_score,candidate_target_index=score_tensor.topk(k=TOPK,dim=1)
candidate_target_score=np.array(candidate_target_score)
candidate_target_index=np.array(candidate_target_index)



c=0
for i in tqdm(range(drug_num)):#针对每一个药物而言
    if np.sum(DTI[i]) == 0:#如果该药物不和任何靶点相互作用，则跳过
        c += 1
        continue
    else:
        drug_interaction_profile=copy.deepcopy(DTI[i])
        TP_FN=np.sum(drug_interaction_profile)#召回率的分母
        predict_target_index=copy.deepcopy(candidate_target_index[i])
        ground_truth_target_index=np.where(drug_interaction_profile==1)[0]
        TP_targets=np.intersect1d(predict_target_index,ground_truth_target_index)
        TP=TP_targets.shape[0]
        temp_recall=TP/TP_FN
        # all_drugs_recall=np.concatenate((all_drugs_recall,temp_recall))
        all_drugs_recall.append(temp_recall)



        # tpr1, fpr1, precision1, recall1 = tpr_fpr_precision_recall(np.array(DTI[i]), np.array(score[i]))

all_drugs_recall=np.array(all_drugs_recall)
average_all_drugs_recall=np.mean(all_drugs_recall)
print("TOP percent:%f TOPK:%d average_recall:%.5f"%(TOPK_precent,TOPK,average_all_drugs_recall))





'''
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
















