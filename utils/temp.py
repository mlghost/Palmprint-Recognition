import pandas as pd

data = pd.read_csv('../extracted_features/HKPU/Merged_HKPU_train.csv')
test = pd.read_csv('../extracted_features/HKPU/Merged_HKPU_test.csv')
# labels = {
#     v: i for i, v in enumerate(set(data['label'].values))
# }
labels = {v: 0 for v in set(data['label'].values)}
i = 0
print labels.keys()
for v in labels.keys():
    labels[v] = i
    i += 1
print sorted(labels.values())
print len(set(sorted(labels.values())))
data['label'] = data['label'].replace(labels)
test['label'] = test['label'].replace(labels)
print data['label']
data.to_csv('Merged_HKPU_train.csv', index=False)
test.to_csv('Merged_HKPU_test.csv', index=False)
print data['label']