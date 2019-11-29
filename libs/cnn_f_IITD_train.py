import numpy as np
import random
import tensorflow as tf
import os
import matplotlib.image as mp
import re
import pandas as pd
from termcolor import colored

from vgg_cnn_f import VGG_CNN_F
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

pic_dim = 224
class_dim = 230


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def get_class(name):
    Chot = np.array([0.0] * class_dim)
    Chot[int(name)] = 1.0
    return Chot


def get_class_number(name):
    res = re.findall('[0-9]+', name)

    return int(res[0]) - 1


batch_size = 10

image = tf.placeholder(tf.float32, [None, 224, 224, 3])
image = tf.div(tf.subtract(image, tf.reduce_mean(image)), reduce_std(image))

labels = tf.placeholder(tf.float32, [None, 230])
net = VGG_CNN_F({'data': image})

fc7 = net.layers['fc7']

print fc7.get_shape()

w1 = tf.Variable(tf.random_normal(shape=[4096, 100], mean=0, stddev=2 / np.sqrt(4096)), name='feature_layer')
b1 = tf.Variable(tf.zeros([100]), name='feature_layer_bias')
output = tf.matmul(fc7, w1) + b1
w = tf.Variable(tf.random_normal(shape=[100, 230], mean=0, stddev=2 / np.sqrt(100)), name='last_W')
b = tf.Variable(tf.zeros([230]), name='last_b')
output1 = tf.matmul(output, w) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output1))
tf.summary.scalar('Cost', cost)
opt = tf.train.AdamOptimizer(learning_rate=0.0001)
train_op = opt.minimize(cost)

merged = tf.summary.merge_all()
saver = tf.train.Saver()
init = tf.initialize_all_variables()

epoch = 40
save_path = './log/weights/cnn_f_IITD_fine_tune.ckpt'
log_path = './log/summaries/IITD/'
extract = True
step = 0
loadw = True
train = False


with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(log_path, sess.graph)
    if loadw:
        saver.restore(sess, save_path)

        print 'Model was restored.'
        im_path = './data/new_right/'
        if extract:
            images = [
                [mp.imread('./data/new_right/' + name), get_class_number(name), name]
                for name in os.listdir('./data/new_right/')
            ]
            features = []
            for i in range(len(images)):
                f = sess.run(output, feed_dict={image: [images[i][0]]})
                features.append(list(f[0]) + [images[i][1]] + [images[i][2]])


            features = pd.DataFrame(features, columns=['vgg_' + str(i) for i in range(100)] + ['vgg_label'] + ['name'])
            features.to_csv('CNNF_features_IITD_train.csv', index=False)

    else:
        sess.run(init)
        net.load('./vgg_cnn_f.npy', sess)
        print "Model's weights were loaded."

    if train:
        images = [
            [mp.imread('./data/new_right/' + name), get_class(get_class_number(name))]
            for name in os.listdir('./data/new_right')
        ]
        for i in range(epoch):
            random.shuffle(images)
            for j in range(len(images) / batch_size):
                xb, yb = [images[p][0] for p in range(j * batch_size, (j + 1) * batch_size)], \
                         [images[p][1] for p in range(j * batch_size, (j + 1) * batch_size)]
                _, c = sess.run([train_op, cost], feed_dict={image: xb, labels: yb})

                if step % 5 == 0:
                    summary = sess.run(merged, feed_dict={image: xb, labels: yb})
                    summary_writer.add_summary(summary, step)
                    print colored('Summaries were updated.', 'magenta')
                if step % 100 == 0:
                    saver.save(sess, save_path)
                    print colored('Model was saved.', 'magenta')
                step += 1
                print colored('Epoch:', 'magenta'), i + 1, '|', colored('Batch:', 'blue'), j + 1, '|', colored('Cost:',
                                                                                                               'green'), c
                print '_____________________________'
        saver.save(sess, save_path)
        print colored('Model was saved.', 'magenta')