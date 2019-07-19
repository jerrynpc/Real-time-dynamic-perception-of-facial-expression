#!/usr/bin/env python 3.6.8
# -*- coding: utf-8 -*-
#Written by jerrynpc 
#写于2019-6月 雄盛科技

from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LSTM
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from keras.backend.tensorflow_backend import set_session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.8 #定量
config.gpu_options.allow_growth = True  #按需
set_session(tf.Session(config=config)) 
tf.ConfigProto()
log_device_placement=True #是否打印设备分配日志
allow_soft_placement=True #如果你指定的设备不存在，允许TF自动分配设备
#tf.reset_default_graph()
########################################################################################
batch_size = 16 # 训练时每个批次的样本数    训练样本数/批次样本数 = 批次数（每个周期）
# num_classes = 10
num_classes = 8 # 表情数据集8个类别
# epochs = 100
epochs = 22 # 训练50周期，训练集所有样本（数据、记录）参与训练一次为一个周期

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'face_emote_trained_model.h5'

img_w = 48
img_h = 48

# LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


model = Sequential()
model.add(Conv2D(256, (3, 3), padding='same',
                 # input_shape=x_train.shape[1:]))
                 input_shape=(48, 48, 3))) # 输入数据是图片转换的矩阵格式，150（行）x 150（列） x 3 （通道）（每个像素点3个单位，分别表示RGB(红绿蓝)）
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())



model.add(Dense(6094))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='mse',
              optimizer=opt,
              metrics=['accuracy'])


# 创建history实例
history = LossHistory()

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.02,
    zoom_range=0.02,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# 训练样本初始化处理:长宽调整，批次大小调整，数据打乱排序（shuffle=True），分类区分（binary：2分类、categorical：多分类）
train_generator = train_datagen.flow_from_directory(
    './dataset/train', # 本例，提供80 x 3 = 240 个训练样本
    target_size=(img_w, img_h),  # 图片格式调整为 150x150
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical')  # matt，多分类

validation_generator = test_datagen.flow_from_directory(
    './dataset/validation',# 本例，提供20 x 3 = 60 个验证样本
    target_size=(img_w, img_h),
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical')  # matt，多分类

#添加tf dashboard
with tf.name_scope('init'): 
    init = tf.global_variables_initializer()
##creat a Session 
sess = tf.Session()
##initialize
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

# 模型适配生成
model.fit_generator(
    train_generator, # 训练集
    samples_per_epoch=33279, # 训练集总样本数，如果提供样本数量不够，则调整图片（翻转、平移等）补足数量（本例，该函数补充2400-240个样本）
    nb_epoch=epochs,
    validation_data=validation_generator, # 验证集
    nb_val_samples=999, # 验证集总样本数，如果提供样本数量不够，则调整图片（翻转、平移等）补足数量（本例，该函数补充800-60个样本）
    callbacks=[history]) # 回调函数，绘制批次（epoch）和精确度（acc）关系图表函数

# Save model and weights
if not os.path.isdir(save_dir): # 没有save_dir对应目录则建立
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

