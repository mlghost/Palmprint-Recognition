import pandas as pd
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

model = SVC(C=1.0, kernel='poly', probability=True)

train = pd.read_csv('../extracted_features/HKPU/HKPU_train.csv')
test = pd.read_csv('../extracted_features/HKPU/HKPU_test.csv')

labels = {v: 0 for v in set(train['label'].values)}
i = 0
print labels.keys()
for v in labels.keys():
    labels[v] = i
    i += 1
print sorted(labels.values())
print len(set(sorted(labels.values())))
train['label'] = train['label'].replace(labels)
test['label'] = test['label'].replace(labels)

X = train[[str(i) for i in range(98)]].values
Y = train['label'].values
print len(set(train['label'].values))

X_test = test[[str(i) for i in range(98)]].values
Y_test = test['label'].values
Y_test = label_binarize(Y_test, classes=[i for i in range(133)])

Y = label_binarize(Y, classes=[i for i in range(133)])
n_classes = Y.shape[1]

random_state = np.random.RandomState(0)
n_samples, n_features = X.shape

classifier = OneVsRestClassifier(svm.SVC(kernel='poly', probability=True,
                                         random_state=random_state))
y_score = classifier.fit(X, Y).decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

lw = 2

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
fig = plt.figure()
plt.plot(fpr["macro"], tpr["macro"],
         color='navy', linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FAR)')
plt.ylabel('Genuine accept rate (GAR)')

plt.grid()
fig.savefig('filename.pdf', format='pdf')
plt.show()
