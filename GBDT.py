from my_function import *
from sklearn.metrics import roc_auc_score
import numpy as np
import lightgbm as lgb
import copy
from sklearn.metrics import auc
import argparse
import random

parser=argparse.ArgumentParser()
parser.add_argument('-f',dest="fold",type=int)
results=parser.parse_args()
fold =results.fold #脚本的第一个参数指示当前运行的是第几折


n = 708#n是药物数据
m = 1512#m是靶点数量
# 读取数据
index_1 = np.loadtxt('divide_result/index_1.txt')#5折交叉验证中1的编号
index_0 = np.loadtxt('divide_result/index_0.txt')#5折交叉验证中0的编号
index = np.hstack((index_1, index_0))
# print("index:",id(index))

# np.savetxt('predict_result_of_ARGA/embedding'+str(fold)+'.txt', embedding)
drug_feature = np.loadtxt('predict_result_of_ARGA/embedding'+str(fold)+'.txt')[0:n, :]#前n行(0到n-1行)是学到的药物的特征向量
target_feature = np.loadtxt('predict_result_of_ARGA/embedding'+str(fold)+'.txt')[n:, :]#后
# drug_feature = max_min_normalize(drug_feature.T).T
# target_feature = max_min_normalize(target_feature.T).T
# label = np.loadtxt('data/DTI.txt')
# label = np.loadtxt("数据集_胡开淼/mat_drug_protein.txt")   # A矩阵  行是药物 列是蛋白质 这个是标签矩阵
label=np.loadtxt('whole_data/DTI_708_1512.txt')
# 获得训练集与测试集的index
idx = copy.deepcopy(index)
# print("idx:",id(idx))
test_index = copy.deepcopy(idx[fold])#将其中的384个1和213714个0作为测试集,测试集有214098个样本
# print("test_index:",id(test_index))
idx=np.delete(idx,fold,axis=0)
train_index = idx.flatten()#flatten()返回的是拷贝 其余四折的1和0都作为训练集，训练集有856392个样本
insersection=np.intersect1d(test_index,train_index)#查看训练集和测试集是否有交集
# print("train_index:",id(train_index))
# 此时test_index和train_index的正负样本分布有一定的聚集性，所以需要打乱
# ！！！注意这里！！！！测试数据不能打乱，训练数据可以打乱
# np.random.seed(10)
# np.random.shuffle(test_index)
np.random.seed(10)
np.random.shuffle(train_index)

# 获得 （测试集 与 训练集） 的 （输入向量 与 标签）
# 此时test_index和train_index都是一维的
test_input, test_output = get_feature(drug_feature, target_feature, label, test_index, n, m)
train_input, train_output = get_feature(drug_feature, target_feature, label, train_index, n, m)#原来不用trick的一行代码
# DTI_WKNKN=np.loadtxt('whole_data/multi_similarity/DTI_708_1512_WKNKN_MAX.txt')#@@@这两行是增加的trick
# train_input, train_output = get_feature_denoise(drug_feature, target_feature, label, train_index, n, m,DTI_WKNKN)#@@@这两行是增加的trick


# 构建SVM| 训练| 预测
print('start training')
lgb_train = lgb.Dataset(train_input, train_output)
lgb_eval = lgb.Dataset(test_input, test_output, reference=lgb_train)
# lightgbm的参数
params = {
    'task': 'train',
    'boosting_type': 'gbdt',#设置提升类型
    'objective': 'binary',#目标二分类这里没问题
    'metric': {'average_precision'},#评估函数
    'is_unbalance': 'true',#针对数据集不平衡的情况进行的优化
    'num_leaves': 80,#叶子节点数
    'learning_rate': 0.02,#学习率
    'feature_fraction': 0.9,# 建树的特征选择比例
    'bagging_fraction': 1,# 建树的样本采样比例
    'bagging_freq': 5,# k 意味着每 k 次迭代执行bagging
    'verbose': 0,#显示模式
    'train_metric':'true',
    "device":"gpu"
}
print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_eval,
                # early_stopping_rounds=2
                )
# y_pred = gbm.predict(test_input, num_iteration=gbm.best_iteration)
y_pred = gbm.predict(test_input, num_iteration=1000)
# t, f, p, r = tpr_fpr_precision_recall(test_output, y_pred)
# print(auc(f, t))
# print(auc(r, p)+r[0]*p[0])
# print('The auc of prediction is:', roc_auc_score(test_output, y_pred))
#
np.savetxt('predict_result_of_ARGA/ARGA' + str(fold) + '.txt', y_pred)
np.savetxt('predict_result_of_ARGA/ARGA_test_index' + str(fold) + '.txt', test_index)
print('end')
    # fold += 1

