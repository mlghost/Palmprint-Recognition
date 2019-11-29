import matplotlib.image as mp
import os
import numpy as np
import random
import pandas as pd

d1 = pd.read_csv('../extracted_features/HKPU/GoogleNet_features_HKPU_train.csv')
d2 = pd.read_csv('../extracted_features/HKPU/CNNF_features_HKPU_train.csv')
print d1
print '_____________________________________________________'
print '_____________________________________________________'
print '_____________________________________________________'
print d2
print '_____________________________________________________'
print '_____________________________________________________'
print '_____________________________________________________'
data = pd.merge(d1,d2,on=['name'])
data = data.drop('name',axis=1)
print data
data.to_csv('Merged_HKPU_train.csv',index=False)