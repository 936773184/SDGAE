import numpy as np
from my_function import *
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve, precision_recall_curve,roc_auc_score,average_precision_score
total_fold=10
tpr_list = []
fpr_list = []
recall_list = []
precision_list = []
for fold in range(total_fold):
    temp_fpr=np.loadtxt('fpr_mean_'+str(fold)+'.txt').tolist()
    temp_tpr=np.loadtxt('tpr_mean_'+str(fold)+'.txt').tolist()
    temp_recall=np.loadtxt('recall_mean_'+str(fold)+'.txt').tolist()
    temp_precision=np.loadtxt('precision_mean_'+str(fold)+'.txt').tolist()
    fpr_list.append(temp_fpr)
    tpr_list.append(temp_tpr)
    recall_list.append(temp_recall)
    precision_list.append(temp_precision)





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
np.savetxt('total_fpr_mean.txt',fpr_mean)
np.savetxt('total_tpr_mean.txt',tpr_mean)
np.savetxt('total_recall_mean.txt',recall_mean)
np.savetxt('total_precision_mean.txt',precision_mean)

print('The auc of prediction is:%.4f'%AUC)
print('The aupr of prediction is:%.4f'%AUPR)

# 画ROC曲线

plt.figure()
plt.plot(fpr_mean,tpr_mean,label='ROC(AUC = %0.4f)' % AUC)
plt.title('ROC curve')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc="lower right")
plt.savefig('ROC_10curves.jpg')
plt.show()

# 画PR曲线

plt.figure()
plt.plot(recall_mean,precision_mean,label='PR(AUPR = %0.4f)' % AUPR)
plt.title('PR curve')
plt.xlabel("RECALL")
plt.ylabel("PRECISION")
plt.legend(loc="lower right")
plt.savefig('PR_10curves.jpg')
plt.show()

with open('metrics.txt', 'w') as f:
    print('AUC:%.6f '%AUC,  'AUPR: %.6f'%AUPR,  file=f)

print('end')







