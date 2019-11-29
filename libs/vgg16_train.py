import numpy as np
import random
import tensorflow as tf
import os
import matplotlib.image as mp
import re
import pandas as pd
from termcolor import colored

from vgg16 import VGG_ILSVRC_16_layers
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

pic_dim = 224
class_dim = 54


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


batch_size = 1

image = tf.placeholder(tf.float32, [None, 224, 224, 3])
image = tf.div(tf.subtract(image, tf.reduce_mean(image)), reduce_std(image))

labels = tf.placeholder(tf.float32, [None, 54])
net = VGG_ILSVRC_16_layers({'data': image})

fc7 = net.layers['fc7']

print fc7.get_shape()


w = tf.Variable(tf.random_normal(shape=[4096, 54], mean=0, stddev=2 / np.sqrt(4096)), name='last_W')
b = tf.Variable(tf.zeros([54]), name='last_b')
output = tf.matmul(fc7, w) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))
tf.summary.scalar('Cost', cost)

prediction = tf.argmax(tf.nn.softmax(output), 1)
equality = tf.equal(prediction, tf.argmax(labels,1))

accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
tf.summary.scalar('Accuracy',accuracy)

opt = tf.train.AdamOptimizer(learning_rate=0.00001)
train_op = opt.minimize(cost)

merged = tf.summary.merge_all()
saver = tf.train.Saver()
init = tf.initialize_all_variables()

epoch = 40
save_path = './log/weights/vgg_16_fine_tune.ckpt'
log_path = './log/summaries/vgg_16/'
extract = False
step = 0
loadw = False
train = True


with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(log_path, sess.graph)
    if loadw:
        saver.restore(sess, save_path)

        print 'Model was restored.'
        im_path = './data/new_images/'
        if extract:
            images = {
                c: {
                    i.split('.')[0]: mp.imread(im_path + c + '/' + i)
                    for i in os.listdir(im_path + c)
                }
                for c in os.listdir(im_path)
            }

            features = []
            for c in images.keys():
                for k in images[c].keys():
                    afs = sess.run(output, feed_dict={image: [images[c][k]]})
                    for af in afs:
                        features.append(list(af) + [c] + [c+ '_'+k])
            features = pd.DataFrame(features, columns=['vgg_' + str(i) for i in range(100)] + ['vgg_label'] + ['name'])
            features.to_csv('VGG16Features.csv', index=False)

    else:
        sess.run(init)
        net.load('./vgg16.npy', sess)
        print "Model's weights were loaded."

    if train:
        images = []
        for folder in os.listdir('./data/new_images/'):
            for instance in os.listdir('./data/new_images/' + folder):
                images.append([mp.imread('./data/new_images/' + folder + '/' + instance), get_class(int(folder))])
        random.shuffle(images)
        train,test = images[:-50],images[-50:]
        for i in range(epoch):
            random.shuffle(train)
            for j in range(len(train) / batch_size):
                xb, yb = [train[p][0] for p in range(j * batch_size, (j + 1) * batch_size)], \
                         [train[p][1] for p in range(j * batch_size, (j + 1) * batch_size)]
                _, c = sess.run([train_op, cost], feed_dict={image: xb, labels: yb})

                if step % 5 == 0:
                    random.shuffle(test)
                    xbt, ybt = [test[p][0] for p in range(5)], \
                             [test[p][1] for p in range(5)]
                    summary = sess.run(merged, feed_dict={image: xbt, labels: ybt})
                    summary_writer.add_summary(summary, step)
                    print colored('Summaries were updated.', 'magenta')
                if step % 10 == 0:
                    saver.save(sess, save_path)
                    print colored('Model was saved.', 'magenta')


                step += 1
                print colored('Epoch:', 'magenta'), i + 1, '|', colored('Batch:', 'blue'), j + 1, '|', colored('Cost:',
                                                                                                               'green'), c
                print '_____________________________'
