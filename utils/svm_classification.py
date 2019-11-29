import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def get_feature(vector, features):
    return [
        f for f, v in zip(features, vector) if v == 1
    ]


def all_zero(col):
    for elem in col:
        if elem != 0:
            return False
    return True


from sklearn.utils import shuffle

google_ppmd_train = pd.read_csv('../extracted_features/PPMD/GoogleNet_features_PPMD_train.csv')
google_ppmd_test = pd.read_csv('../extracted_features/PPMD/GoogleNet_features_PPMD_test.csv')

google_iitd_train = pd.read_csv('../extracted_features/IITD/GoogleNet_features_IITD_train.csv')
google_iitd_test = pd.read_csv('../extracted_features/IITD/GoogleNet_features_IITD_test.csv')

google_hkpu_train = pd.read_csv('../extracted_features/HKPU/GoogleNet_features_HKPU_train.csv')
google_hkpu_test = pd.read_csv('../extracted_features/HKPU/GoogleNet_features_HKPU_test.csv')

cnnf_ppmd_train = pd.read_csv('../extracted_features/PPMD/CNNF_features_PPMD_train.csv')
cnnf_ppmd_test = pd.read_csv('../extracted_features/PPMD/CNNF_features_PPMD_test.csv')

cnnf_iitd_train = pd.read_csv('../extracted_features/IITD/CNNF_features_IITD_train.csv')
cnnf_iitd_test = pd.read_csv('../extracted_features/IITD/CNNF_features_IITD_test.csv')

cnnf_hkpu_train = pd.read_csv('../extracted_features/HKPU/CNNF_features_HKPU_train.csv')
cnnf_hkpu_test = pd.read_csv('../extracted_features/HKPU/CNNF_features_HKPU_test.csv')

cnnf_hkpu_train = shuffle(cnnf_hkpu_train)
sc = []
for n in range(1, 16):
    train = []
    test = []
    for i in range(386):
        ds = cnnf_hkpu_train[
            cnnf_hkpu_train['vgg_label'] == i]
        dtr = ds.iloc[:n]
        for value in dtr.values:
            train.append(value)
    train = pd.DataFrame(data=train, columns=['vgg_' + str(i) for i in range(100)]+ ['vgg_label'] +['name'])

    X = train[['vgg_' + str(i) for i in range(100)]].values
    Y = train['vgg_label'].values
    X_test = cnnf_hkpu_test[['vgg_' + str(i) for i in range(100)]].values
    Y_test = cnnf_hkpu_test['vgg_label'].values

    model = SVC(C=1.0, kernel='poly', max_iter=4000)

    model.fit(X, Y)
    sc.append(accuracy_score(Y_test, model.predict(X_test)))
    print 'Number of Training Samples:', n
    print 'Accuracy:', str(accuracy_score(Y_test, model.predict(X_test))) + '%'

"""
GoogleNet IITD
    Number of Training Samples: 1
    Accuracy: 0.3869565217391304%
    Number of Training Samples: 2
    Accuracy: 0.5956521739130435%
    Number of Training Samples: 3
    Accuracy: 0.7%
    Number of Training Samples: 4
    Accuracy: 0.7652173913043478%    
    Number of Training Samples: 5
    Accuracy: 0.7739130434782608%
----------------------------------------
GoogleNet HKPU
     Number of Training Samples: 1
     Accuracy: 0.27461139896373055%
     Number of Training Samples: 2
     Accuracy: 0.4637305699481865%
     Number of Training Samples: 3
     Accuracy: 0.5595854922279793%
     Number of Training Samples: 4
     Accuracy: 0.6062176165803109%
     Number of Training Samples: 5
     Accuracy: 0.6321243523316062%
     Number of Training Samples: 6
     Accuracy: 0.6580310880829016%
     Number of Training Samples: 7
     Accuracy: 0.6735751295336787%
     Number of Training Samples: 8
     Accuracy: 0.6865284974093264%
     Number of Training Samples: 9
     Accuracy: 0.6917098445595855%
     Number of Training Samples: 10
     Accuracy: 0.6968911917098446%
     Number of Training Samples: 11
     Accuracy: 0.6994818652849741%
     Number of Training Samples: 12
     Accuracy: 0.7202072538860104%
     Number of Training Samples: 13
     Accuracy: 0.7305699481865285%
     Number of Training Samples: 14
     Accuracy: 0.7435233160621761%
     Number of Training Samples: 15
     Accuracy: 0.7435233160621761%
-----------------------------------------    
GoogleNetPPMD
    Number of Training Samples: 1
    Accuracy: 0.5185185185185185%
    Number of Training Samples: 2
    Accuracy: 0.6851851851851852%
    Number of Training Samples: 3
    Accuracy: 0.8148148148148148%
    Number of Training Samples: 4
    Accuracy: 0.8703703703703703%
    Number of Training Samples: 5
    Accuracy: 0.9259259259259259%
-----------------------------------------
CNNF HKPU
    Number of Training Samples: 1
    Accuracy: 0.3963730569948187%
    Number of Training Samples: 2
    Accuracy: 0.5207253886010362%
    Number of Training Samples: 3
    Accuracy: 0.5777202072538861%
    Number of Training Samples: 4
    Accuracy: 0.6088082901554405%
    Number of Training Samples: 5
    Accuracy: 0.6450777202072538%
    Number of Training Samples: 6
    Accuracy: 0.6632124352331606%
    Number of Training Samples: 7
    Accuracy: 0.689119170984456%
    Number of Training Samples: 8
    Accuracy: 0.6968911917098446%
    Number of Training Samples: 9
    Accuracy: 0.7046632124352331%
    Number of Training Samples: 10
    Accuracy: 0.7150259067357513%
    Number of Training Samples: 11
    Accuracy: 0.7176165803108808%
    Number of Training Samples: 12
    Accuracy: 0.7202072538860104%
    Number of Training Samples: 13
    Accuracy: 0.7227979274611399%
    Number of Training Samples: 14
    Accuracy: 0.7331606217616581%
    Number of Training Samples: 15
    Accuracy: 0.7383419689119171%
--------------------------------------
CNNF PPMD

    Number of Training Samples: 1
    Accuracy: 0.7222222222222222%
    Number of Training Samples: 2
    Accuracy: 0.8333333333333334%
    Number of Training Samples: 3
    Accuracy: 0.9444444444444444%
    Number of Training Samples: 4
    Accuracy: 0.9444444444444444%
----------------------------------------    

CNNF IITD 
    Number of Training Samples: 1
    Accuracy: 0.5782608695652174%
    Number of Training Samples: 2
    Accuracy: 0.7347826086956522%
    Number of Training Samples: 3
    Accuracy: 0.7956521739130434%
    Number of Training Samples: 4
    Accuracy: 0.8173913043478261%
----------------------------------------
"""
