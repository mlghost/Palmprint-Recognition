import pandas as pd
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# data1 = pd.read_csv('../extracted_features/PPMD/PPMD_CONV2D_train.csv')
# data1 = shuffle(data1)
# X, y = data1[[str(i) for i in range(24)]].values, data1['label'].values
# y = label_binarize(y, classes=[i for i in range(54)])
# df_test = pd.read_csv('../extracted_features/PPMD/PPMD_CONV2D_test.csv')
# x_test, y_test = df_test[[str(i) for i in range(24)]].values, df_test['label'].values
#
# y_test = label_binarize(y_test, classes=[i for i in range(54)])
#
# n_classes = 54
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
#
# classifier = OneVsRestClassifier(svm.SVC(kernel='poly', probability=True,
#                                          random_state=random_state))
# y_score = classifier.fit(X, y).decision_function(x_test)
#
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
#
# ind = []
# for i in range(54):
#     fpr[i], tpr[i], threshold = roc_curve(y_test[:, i], y_score[:, i])
#     if not np.isnan(tpr[i][0]):
#         print 'yes'
#
#         roc_auc[i] = auc(fpr[i], tpr[i])
#         ind.append(i)
#     else:
#         print 'no'
# lw = 2
#
# all_fpr = np.unique(np.concatenate([fpr[i] for i in ind]))
#
# mean_tpr = np.zeros_like(all_fpr)
# for i in ind:
#     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#
# mean_tpr /= len(ind)
#
# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
#
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
# plt.plot(fpr["macro"], tpr["macro"],
#          color='green', linestyle='solid', linewidth=2)
# print 'PPMD:'
# print ("SVM auc", roc_auc["macro"])
# fnr = 1 - mean_tpr
# EER = all_fpr[np.nanargmin(np.absolute((fnr - all_fpr)))]
# print 'SVM EER:', EER
# print np.mean(EER)
#
# ________________________________________________
#
data1 = pd.read_csv('../extracted_features/IITD/IITD_CONV2D_train.csv')
data1 = shuffle(data1)
X, y = data1[[str(i) for i in range(49)]].values, data1['label'].values
y = label_binarize(y, classes=[i for i in range(230)])
df_test = pd.read_csv('../extracted_features/IITD/IITD_CONV2D_test.csv')
x_test, y_test = df_test[[str(i) for i in range(49)]].values, df_test['label'].values
y_test = label_binarize(y_test, classes=[i for i in range(230)])

n_classes = 230
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape

classifier = OneVsRestClassifier(svm.SVC(kernel='poly', probability=True,
                                         random_state=random_state))
y_score = classifier.fit(X, y).decision_function(x_test)

fpr = dict()
tpr = dict()
roc_auc = dict()

ind = []
for i in range(230):
    fpr[i], tpr[i], threshold = roc_curve(y_test[:, i], y_score[:, i])
    if not np.isnan(tpr[i][0]):
        print 'yes'

        roc_auc[i] = auc(fpr[i], tpr[i])
        ind.append(i)
    else:
        print 'no'
lw = 2

all_fpr = np.unique(np.concatenate([fpr[i] for i in ind]))

mean_tpr = np.zeros_like(all_fpr)
for i in ind:
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= len(ind)

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
plt.plot(fpr["macro"], tpr["macro"],
         color='blue', linestyle='solid', linewidth=1)
print ''
print "SVM auc", roc_auc["macro"]
fnr = 1 - mean_tpr
EER = all_fpr[np.nanargmin(np.absolute((fnr - all_fpr)))]
print 'SVM EER:', EER
print np.mean(EER)
# ___________________________________________________
#
# data1 = pd.read_csv('../extracted_features/HKPU/HKPU_CONV2D_train.csv')
# df_test = pd.read_csv('../extracted_features/HKPU/HKPU_CONV2D_test.csv')
#
#
# labels = {v: 0 for v in set(data1['label'].values)}
# i = 0
# print labels.keys()
# for v in labels.keys():
#     labels[v] = i
#     i += 1
# print sorted(labels.values())
# print len(set(sorted(labels.values())))
# data1['label'] = data1['label'].replace(labels)
# df_test['label'] = df_test['label'].replace(labels)
#
#
# data1 = shuffle(data1)
# X, y = data1[[str(i) for i in range(49)]].values, data1['label'].values
# print data1['label'].unique()
# y = label_binarize(y, classes=[i for i in range(len(set(data1['label'].values)))])
#
#
# x_test, y_test = df_test[[str(i) for i in range(49)]].values, df_test['label'].values
# y_test = label_binarize(y_test, classes=[i for i in range(len(set(data1['label'].values)))])
#
# n_classes = len(set(data1['label'].values))
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
#
# classifier = OneVsRestClassifier(svm.SVC(kernel='poly', probability=True,
#                                          random_state=random_state))
# y_score = classifier.fit(X, y).decision_function(x_test)
#
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# c = 0
# ind = []
# for i in range(len(set(data1['label'].values))):
#     fpr[i], tpr[i], threshold = roc_curve(y_test[:, i], y_score[:, i])
#     if not np.isnan(tpr[i][0]):
#         print 'yes'
#
#         roc_auc[i] = auc(fpr[i], tpr[i])
#         ind.append(i)
#     else:
#         print 'no'
# lw = 2
#
# all_fpr = np.unique(np.concatenate([fpr[i] for i in ind]))
#
# mean_tpr = np.zeros_like(all_fpr)
# for i in ind:
#     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#
# mean_tpr /= len(ind)
#
# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# print 'HKPU'
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
# plt.plot(fpr["macro"], tpr["macro"],
#          color='red', linestyle='solid', linewidth=1)
# print "SVM auc", roc_auc["macro"]
# fnr = 1 - mean_tpr
# EER = all_fpr[np.nanargmin(np.absolute((fnr - all_fpr)))]
# print 'SVM EER:', EER
# print np.mean(EER)
#
# ----------------------------------------------
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Acceptance Rate (FAR)')
plt.ylabel('Genuine Acceptance Rate (GAR)')
# plt.legend(['SMPD', 'IITD', 'HKPU'])
plt.grid()
# fig.savefig('ROC_3.pdf', format='pdf')
plt.show()
