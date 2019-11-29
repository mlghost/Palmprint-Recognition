from scipy.misc import imresize
import matplotlib.image as mp
import os
import matplotlib.pyplot as plt
import numpy as np
# dir = ['../data/Right/' + name for name in os.listdir('../data/Right')]
# p = 0
# size = 224
# for d in dir:
#     images = [imresize(mp.imread(d + '/' + name), (size, size)) for name in os.listdir(d)]
#     os.makedirs('../data/new_right/' + str(p))
#     imname = 0
#     for image in images:
#         print(image.shape)
#         mp.imsave('../data/new_right/' + str(p) + '/' + str(imname) + '.JPG', image, format='JPG')
#         imname += 1
#     print(p)
#     p += 1


dir = [name[:name.index('.')] for name in os.listdir('../data/hkpu')]
size = 224
for image_name in dir:
    image = imresize(mp.imread('../data/hkpu/'+ image_name + '.jpg'), (size, size))
    mp.imsave('../data/gray_hkpu/'+ image_name + '.JPG', image, format='JPG',cmap='gray')
