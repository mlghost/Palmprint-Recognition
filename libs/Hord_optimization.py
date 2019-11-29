import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from termcolor import colored
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

CONV_KERNEL_LENGTH = 3
# CLASS_DIM = 54
d1 = pd.read_csv('../extracted_features/HKPU/Merged_HKPU_train.csv')
X, y = d1[[str(i) for i in range(100)] + ['vgg_' + str(i) for i in range(100)]].values, d1[
    'label'].values
d2 = pd.read_csv('../extracted_features/HKPU/Merged_HKPU_test.csv')
x_test, y_test = d2[[str(i) for i in range(100)] + ['vgg_' + str(i) for i in range(100)]].values, \
                 d2['label'].values
print d2['label'].max()
print d1['label'].max()

def evaluate(xtrain, xtest, y_train, y_test):
    model = SVC(max_iter=2000)
    model.fit(xtrain, y_train)
    return accuracy_score(y_test, model.predict(xtest))


def conv_1d(input, length, kernel, klen, stride=2):
    ex_data = []
    if (length - klen) % stride != 0:
        for i in range(len(input)):
            data = np.array(np.zeros(length + 1))
            data[:length] = input[i]
            ex_data.append(data)
        L = length + 1
        ex_data = np.array(ex_data)
    else:
        ex_data = input
        L = length
    t_output = []
    for j in range(len(ex_data)):
        output = []
        for i in range(0, int((L - klen)), stride):
            output.append(np.dot(ex_data[j][i: i + klen], kernel))
        t_output.append(output)
    return np.array(t_output)


def conv_2d(input, length, kernel, klen, stride=2):
    dataShape = list(np.shape(input))

    ex_data = []
    if (length - klen) % stride != 0:
        for i in range(len(input)):
            data = np.array(np.zeros((dataShape[1], dataShape[-1] + 1)))
            data[:, :length] = input[i, :]
            ex_data.append(data)
        L = length + 1
        ex_data = np.array(ex_data)
    else:
        ex_data = input
        L = length
    t_output = []
    for j in range(len(ex_data)):
        output = []
        for i in range(0, int((L - klen)), stride):
            output.append(np.sum(ex_data[j][:, i: i + klen] * kernel))
        t_output.append(output)
    return np.array(t_output)


class ConvolutionKernelOptimization:
    def __init__(self, dim=CONV_KERNEL_LENGTH):
        self.xlow = -1 * np.ones(dim)
        self.xup = 1 * np.ones(dim)
        self.dim = dim
        self.info = 'Kernel Length= {}'.format('( ' + '2,' + str(5) + ' )')
        self.integer = []
        self.continuous = np.arange(0, dim)

    def objfunction(self, kernel):
        # kernel = np.reshape(kernel, (2, int(self.dim / 2)))
        xtrain = conv_1d(X, 200, kernel, CONV_KERNEL_LENGTH)
        print np.shape(xtrain)
        xtest = conv_1d(x_test, 200, kernel, CONV_KERNEL_LENGTH)
        c = evaluate(xtrain, xtest, y, y_test)
        print 'Accuracy:', colored(c, 'green')
        print kernel
        print '_______________________________'
        return 1 - c


# ________________________________________________________
# ________________________________________________________
# ________________________________________________________
# ________________________________________________________


# from pySOT import *
# from poap.controller import SerialController
#
# maxeval = 1000
# data = ConvolutionKernelOptimization(dim=CONV_KERNEL_LENGTH)
# print(data.info)
#
# controller = SerialController(data.objfunction)
# controller.strategy = \
#     SyncStrategyNoConstraints(
#         worker_id=0, data=data,
#         maxeval=maxeval, nsamples=1,
#         exp_design=LatinHypercube(dim=data.dim, npts=2 * (data.dim + 1)),
#         response_surface=RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=maxeval),
#         sampling_method=CandidateDYCORS(data=data, numcand=1000 * data.dim))
#
# result = controller.run()
#
# print('Best value found: {0}'.format(1 - result.value))
# print('Best solution found: {0}'.format(
#     np.array_str(result.params[0], max_line_width=np.inf,
#                  precision=5, suppress_small=True)))
# import matplotlib.pyplot as plt
#
# fvals = np.array([o.value for o in controller.fevals])
#
# f, ax = plt.subplots()
# print fvals
# plt.scatter(np.arange(0, maxeval), fvals, s=40, cmap='g', edgecolors='None', alpha=0.6)  # Points
# plt.plot(np.arange(0, maxeval), np.minimum.accumulate(fvals), 'r-', linewidth=1.0)  # Best value found
# plt.xlabel('Step')
# plt.ylabel('Error Value')
# plt.title(data.info)
# plt.show()

xtrain = conv_1d(X, 200, [-0.00953077, -0.08742015, -0.35574549], CONV_KERNEL_LENGTH)
df_train = pd.DataFrame(data=xtrain, columns=[str(i) for i in range(99)])
df_train['label'] = d1['label'].values
print d1['label']
xtest = conv_1d(x_test, 200, [-0.00953077, -0.08742015, -0.35574549], CONV_KERNEL_LENGTH)
df_test = pd.DataFrame(data=xtest, columns=[str(i) for i in range(99)])
df_test['label'] = d2['label'].values

print df_train
#
df_train.to_csv('HKPU_train.csv', index=False)
df_test.to_csv('HKPU_test.csv', index=False)
#
"""
Merged PPMD Conv1D
|      kernel     |     Stride     |     Accuracy     |    Number Of Features
         3                2                 100                    99
         5                2                 100                    98
         5                3                 100                    65
         7                3                 100                    65
-----------------------------------------------------------------------------
Merged HKPU Conv1D
|      kernel     |     Stride     |     Accuracy     |    Number Of Features
         3                2                 95.8                   99
         5                2                 95.8                   98
         5                3                 94.5                   65
         7                3                 94.8                   65         
-----------------------------------------------------------------------------
Merged IITD Conv1D
|      kernel     |     Stride     |     Accuracy     |    Number Of Features
         3                2                 99.5                   99
         5                2                 99.5                   98
         5                3                 99.1                   65
         7                3                 98.6                   65

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
Merged PPMD Conv2D
|      kernel     |     Stride     |     Accuracy     |    Number Of Features    
         3                2                 100                    49
         5                2                 100                    33
         5                3                 100                    32
         5                4                 100                    19
-----------------------------------------------------------------------------
Merged IITD Conv2D
|      kernel     |     Stride     |     Accuracy     |    Number Of Features          
         3                2                 99.1                   49
         3                3                 98.3                   33
         5                3                 98.2                   32
         5                5                 94.7                   19
-----------------------------------------------------------------------------
Merged HKPU Conv2D
|      kernel     |     Stride     |     Accuracy     |    Number Of Features          
         3                2                 96.9                   49
         3                3                 93.9                   33
         5                3                 93.7                   32
         5                5                 90.1                   19

"""
