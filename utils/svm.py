import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train = pd.read_csv('../extracted_features/PPMD/GoogleNet_features_PPMD_train.csv')
test = pd.read_csv('../extracted_features/PPMD/GoogleNet_features_PPMD_test.csv')
model = SVC(max_iter=2000, kernel='poly')
x_train, y_train = train[[str(i) for i in range(100)]].values, train['label'].values
x_test, y_test = test[[str(i) for i in range(100)]].values, test['label'].values
model.fit(x_train, y_train)
print accuracy_score(y_test, model.predict(x_test))
