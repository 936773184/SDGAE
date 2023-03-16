from numpy import *;
import numpy as np;
from math import *;


def null_list(num):
    lis = []
    for i in range(num): lis.append([])
    return lis


def laplacians(A):
    n = A.shape[0]                # m = n = 2220
    m = A.shape[1]
    A1 = A + np.eye(A.shape[0]) #为邻接矩阵加自连
    D = np.sum(A1, axis=1)         # 计算每一行的和 D.shape=2220,1 其中的每一个元素代表结点的度
    A_L = np.zeros(A.shape)
    for i in range(n):
        for j in range(m):
            if i == j and D[i] != 0:
                A_L[i, j] = 1
            elif i != j and A1[i, j] != 0:
                A_L[i, j] = (-1)/sqrt(D[i] * D[j])
            else:
                A_L[i, j] = 0
    return A_L


def max_min_normalize(a):                              #矩阵归一化
    sum_of_line = np.sum(a, axis=1)
    line = a.shape[0]
    row = a.shape[1]
    i = 0
    while i < line:
        j = 0
        while j < row:
            if sum_of_line[i] != 0:
                max = np.max(a[i])
                min = np.min(a[i])
                a[i, j] = (a[i, j]-min) / (max-min)
            j = j + 1
        i = i + 1
    return a


def equal_len_list(a):      # 孙畅按比例采样
    row_len = []
    for i in a:
        row_len.append(len(i))
    min_len = min(row_len)
    equal_len_a = []
    for i in a:
        tem_list = []
        multi = len(i)/min_len
        for j in range(min_len):
            tem_list.append(i[int(j*multi)])
        equal_len_a.append(tem_list)
    return equal_len_a


def remove_ele(a, x):
    b = []
    for i in a:
        flag = 1
        while flag == 1:
            if x in i:
                i.remove(x)
            else:
                b.append(i)
                flag = 0
    return b


def get_feature(drug_feature, target_feature, label, index, drug_num, target_num):
    input = []
    output = []
    for i in range(index.shape[0]):
        drug = int(index[i] / target_num)
        target = int(index[i] % target_num)
        feature = np.hstack((drug_feature[drug], target_feature[target]))
        input.append(feature.tolist())
        output.append(label[drug, target])
    return np.array(input), np.array(output)
def get_feature_denoise(drug_feature, target_feature, label, index, drug_num, target_num,DTI_WKNKN):#根据WKNKN后的DTI矩阵查找，如果WKNKN后的DTI矩阵中的药物靶点对是小数，我们就不加入训练
    input = []
    output = []
    for i in range(index.shape[0]):
        drug = int(index[i] / target_num)
        target = int(index[i] % target_num)
        if (DTI_WKNKN[drug][target] != 0) and (DTI_WKNKN[drug][target] != 1): #如果在WKNKN矩阵中的对应值不是0或1，那么我们就不加入训练
            continue
        feature = np.hstack((drug_feature[drug], target_feature[target]))
        input.append(feature.tolist())
        output.append(label[drug, target])
    return np.array(input), np.array(output)

def line_normalize(A):                        #行归一化
    sum_of_line = np.sum(A, axis=1)
    line = A.shape[0]
    row = A.shape[1]
    i = 0
    while i < line:
        j = 0
        while j < row:
            if sum_of_line[i] != 0:
                A[i, j] = A[i, j] / sum_of_line[i]
            j = j + 1
        i = i + 1
    return A

# true是标签，pred是预测值
def tpr_fpr_precision_recall(true, pred):
    num = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    index = list(reversed(np.argsort(pred)))
    tpr = []
    fpr = []
    precision = []
    recall = []
    for i in range(pred.shape[0]):
        if true[int(index[i])] == 1:
            tp += 1
        else:
            fp += 1
        if np.sum(true) == 0:
            tpr.append(0)
            fpr.append(0)
            precision.append(0)
            recall.append(0)
        else:
            tpr.append(tp / np.sum(true))
            fpr.append(fp / (true.shape[0] - np.sum(true)))
            precision.append(tp / (tp + fp))
            recall.append(tp / np.sum(true))
    return tpr, fpr, precision, recall
