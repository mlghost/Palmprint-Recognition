from matplotlib import rcParams
import matplotlib.pyplot as plt

# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Tahoma']

plt.rcParams["font.family"] = "Times New Roman"

import pandas as pd


def smooth(y, gamma,name=''):
    yp = [y[0]]
    for i in range(1, len(y)):
        yp.append(gamma * y[i] + (1 - gamma) * yp[-1])
    print name,yp[-1]
    return yp


data = pd.read_csv('../cost_acc_result/hkpu-CNNF.csv')
data1 = pd.read_csv('../cost_acc_result/hkpu-GOOGLE.csv')
# data2 = pd.read_csv('vgg19acc.csv')
# data3 = pd.read_csv('cnnfacc.csv')

max = min([data['Step'].max(),data1['Step'].max()])#960056
# data = data[data['Step'] <= max]

plt.plot(100 *data[data['Step'] <= max]['Step'].values, smooth(data[data['Step'] <= max]['Value'].values,.07,name='vgg16'), linewidth=1, c='#fcb001')
plt.plot(100 *data1[data1['Step'] <= max]['Step'].values, smooth(data1[data1['Step'] <= max]['Value'].values,.3,name='googlenet'), linewidth=1, c='#001146')
# plt.plot(100 *data2[data2['Step'] <= max]['Step'].values, smooth(data2[data2['Step'] <= max]['Value'].values,.25,name='vgg19'), linewidth=1, c='#05696b')
# plt.plot(100 *data3[data3['Step'] <= max]['Step'].values, smooth(data3[data3['Step'] <= max]['Value'].values,.2,name='cnnf'), linewidth=1, c='#8ffe09')



# plt.plot(100 * data[data['Step'] <= max]['Step'].values, smooth(data[data['Step'] <= max]['Value'].values,.1), linewidth=1, c='#fcb001')
# plt.plot(100 * data1[data1['Step'] <= max]['Step'].values, smooth(data1[data1['Step'] <= max]['Value'].values,.1), linewidth=1, c='#001146')
# plt.plot(100 * data2[data2['Step'] <= max]['Step'].values, smooth(data2[data2['Step'] <= max]['Value'].values,.1), linewidth=1, c='#05696b')
# plt.plot(100 * data3[data3['Step'] <= max]['Step'].values, smooth(data3[data3['Step'] <= max]['Value'].values,.1), linewidth=1, c='#8ffe09')

plt.grid(True)
plt.xlabel('Step')
plt.ylabel('Classification Cost')
plt.legend(['CNN-F','GoogleNet'])
plt.show()
