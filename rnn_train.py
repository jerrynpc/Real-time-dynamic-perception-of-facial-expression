
#Written by jerrynpc 
#写于2019-6月 雄盛科技


from __future__ import print_function
import tensorflow as tf
from time import time
import numpy as np
from LSTM.setting import batch_size, width, height, rnn_size, out_size, channel, learning_rate, num_epoch

#########################################

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
# 指定第一块GPU可用 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #指定GPU的第二种方法
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.8 #定量
config.gpu_options.allow_growth = True  #按需
set_session(tf.Session(config=config)) 
#tf.ConfigProto()
#log_device_placement=True #是否打印设备分配日志
#allow_soft_placement=True #如果你指定的设备不存在，允许TF自动分配设备
'''
训练主函数
tensorboard --logdir=./
'''

def weight_variable(shape, w_alpha=0.01):
    '''
    增加噪音，随机生成权重
    :param shape: 权重形状
    :param w_alpha:随机噪声
    :return:
    '''
    initial = w_alpha * tf.random_normal(shape)
    return tf.Variable(initial)
def bias_variable(shape, b_alpha=0.1):
    '''
    增加噪音，随机生成偏置项
    :param shape:权重形状
    :param b_alpha:随机噪声
    :return:
    '''
    initial = b_alpha * tf.random_normal(shape)
    return tf.Variable(initial)
def rnn_graph(x, rnn_size, out_size, width, height, channel):
    '''
    循环神经网络计算图
    :param x:输入数据
    :param rnn_size:
    :param out_size:
    :param width:
    :param height:
    :return:
    '''
    # 权重及偏置
    w = weight_variable([rnn_size, out_size])
    b = bias_variable([out_size])
    # LSTM
    # rnn_size这里指BasicLSTMCell的num_units，指输出的向量维度
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
    # transpose的作用将(?,32,448,3)形状转为(32,?,448,3),?为batch-size，32为高，448为宽，3为通道数（彩图）
    # 准备划分为32个相同网络，输入序列为(448,3)，这样速度较快，逻辑较为符合一般思维
    x = tf.transpose(x, [1,0,2,3])
    # reshape -1 代表自适应，这里按照图像每一列的长度为reshape后的列长度
    x = tf.reshape(x, [-1, channel*width])
    # split默任在第一维即0 dimension进行分割，分割成height份，这里实际指把所有图片向量按对应行号进行重组
    x = tf.split(x, height)
    # 这里RNN会有与输入层相同数量的输出层，我们只需要最后一个输出
    outputs, status = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
    y_conv = tf.add(tf.matmul(outputs[-1], w), b)
    return y_conv

def accuracy_graph(y, y_conv):
    '''
    偏差计算图
    :param y:
    :param y_conv:
    :return:
    '''
    correct = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy

def get_batch(image_list,label_list,img_width,img_height,batch_size,capacity,channel):
    '''
    #通过读取列表来载入批量图片及标签
    :param image_list: 图片路径list
    :param label_list: 标签list
    :param img_width: 图片宽度
    :param img_height: 图片高度
    :param batch_size:
    :param capacity:
    :return:
    '''
    image = tf.cast(image_list,tf.string)
    label = tf.cast(label_list,tf.int32)
    input_queue = tf.train.slice_input_producer([image,label],shuffle=True)
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])

    image = tf.image.decode_jpeg(image_contents,channels=channel)
    image = tf.cast(image,tf.float32)
    if channel==3:
        image -= [42.79902,42.79902,42.79902] # 减均值
    elif channel == 1:
        image -= 42.79902  # 减均值
    image.set_shape((img_height,img_width,channel))
    image_batch,label_batch = tf.train.batch([image,label],batch_size=batch_size,num_threads=64,capacity=capacity)
    label_batch = tf.reshape(label_batch,[batch_size])

    return image_batch,label_batch

def get_file(file_dir):
    '''
    通过文件路径获取图片路径及标签
    :param file_dir: 文件路径
    :return:
    '''
    images = []
    for root,sub_folders,files in os.walk(file_dir):
        for name in files:
            images.append(os.path.join(root,name))
    labels = []
    for label_name in images:
        letter = label_name.split("\\")[-2]
        if letter =="anger":labels.append(00)
        elif letter =="contempt":labels.append(11)
        elif letter == "disgust":labels.append(22)
        elif letter == "fear":labels.append(33)
        elif letter == "happy":labels.append(44)
        elif letter == "normal":labels.append(55)
        elif letter == "sad":labels.append(66)
        elif letter == "surprised":labels.append(77)
        labels.append(8)

    print("check for get_file:",images[0],"label is ",labels[0])
    #shuffle
    temp = np.array([images,labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(float(i)) for i in label_list]
    return image_list,label_list

#标签格式重构
def onehot(labels):
    n_sample = len(labels)
    n_class = 9  # max(labels) + 1
    onehot_labels = np.zeros((n_sample,n_class))
    onehot_labels[np.arange(n_sample),labels] = 1
    return onehot_labels

if __name__ == '__main__':
    startTime = time()
    # 按照图片大小申请占位符
    x = tf.placeholder(tf.float32, [None, height, width, channel])
    y = tf.placeholder(tf.float32)
    # rnn模型
    y_conv = rnn_graph(x, rnn_size, out_size, width, height, channel)
    # 独热编码转化
    y_conv_prediction = tf.argmax(y_conv, 1)
    y_real = tf.argmax(y, 1)
    # 优化计算图
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # 偏差
    accuracy = accuracy_graph(y, y_conv)
    # 自训练图像
    xs, ys = get_file('./dataset/train/')  # 获取图像列表与标签列表
    image_batch, label_batch = get_batch(xs, ys, img_width=width, img_height=height, batch_size=batch_size, capacity=256,channel=channel)
    # 验证集
    xs_val, ys_val = get_file('./dataset/validation/')  # 获取图像列表与标签列表
    image_val_batch, label_val_batch = get_batch(xs_val, ys_val, img_width=width, img_height=height,batch_size=455, capacity=256,channel=channel)
    # 启动会话.开始训练
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # 启动线程
    coord = tf.train.Coordinator()  # 使用协调器管理线程
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    # 日志记录
    summary_writer = tf.summary.FileWriter('./logs/', graph=sess.graph, flush_secs=15)
    summary_writer2 = tf.summary.FileWriter('./logs/plot2/', flush_secs=15)
    tf.summary.scalar(name='loss_func', tensor=loss)
    tf.summary.scalar(name='accuracy', tensor=accuracy)
    merged_summary_op = tf.summary.merge_all()

    step = 0
    acc_rate = 0.98
    epoch_start_time = time()
    for i in range(num_epoch):
        batch_x, batch_y = sess.run([image_batch, label_batch])
        batch_y = onehot(batch_y)

        merged_summary,_,loss_show = sess.run([merged_summary_op,optimizer,loss], feed_dict={x: batch_x, y: batch_y})
        summary_writer.add_summary(merged_summary, global_step=i)

        if i % (int(7000//batch_size)) == 0:
            batch_x_test, batch_y_test = sess.run([image_val_batch, label_val_batch])
            batch_y_test = onehot(batch_y_test)
            batch_x_test = batch_x_test.reshape([-1, height, width, channel])
            merged_summary_val,acc,prediction_val_out,real_val_out,loss_show = sess.run([merged_summary_op,accuracy,y_conv_prediction,y_real,loss],feed_dict={x: batch_x_test, y: batch_y_test})
            summary_writer2.add_summary(merged_summary_val, global_step=i)

            # 输出每个类别正确率
            lh1_right, lh2_right, lh3_right, lh4_right, lh5_right, lh6_right, lh7_right = 0, 0, 0, 0, 0, 0, 0
            lh1_wrong, lh2_wrong, lh3_wrong, lh4_wrong, lh5_wrong, lh6_wrong, lh7_wrong = 0, 0, 0, 0, 0, 0, 0
            for ii in range(len(prediction_val_out)):
                if prediction_val_out[ii] == real_val_out[ii]:
                    if real_val_out[ii] == 0:
                        lh1_right += 1
                    elif real_val_out[ii] == 1:
                        lh2_right += 1
                    elif real_val_out[ii] == 2:
                        lh3_right += 1
                    elif real_val_out[ii] == 3:
                        lh4_right += 1
                    elif real_val_out[ii] == 4:
                        lh5_right += 1
                    elif real_val_out[ii] == 5:
                        lh6_right += 1
                    elif real_val_out[ii] == 6:
                        lh7_right += 1
                else:
                    if real_val_out[ii] == 0:
                        lh1_wrong += 1
                    elif real_val_out[ii] == 1:
                        lh2_wrong += 1
                    elif real_val_out[ii] == 2:
                        lh3_wrong += 1
                    elif real_val_out[ii] == 3:
                        lh4_wrong += 1
                    elif real_val_out[ii] == 4:
                        lh5_wrong += 1
                    elif real_val_out[ii] == 5:
                        lh6_wrong += 1
                    elif real_val_out[ii] == 6:
                        lh7_wrong += 1
            print(step, "correct rate :", ((lh1_right) / (lh1_right + lh1_wrong)), ((lh2_right) / (lh2_right + lh2_wrong)),
                  ((lh3_right) / (lh3_right + lh3_wrong)), ((lh4_right) / (lh4_right + lh4_wrong)),
                  ((lh5_right) / (lh5_right + lh5_wrong)), ((lh6_right) / (lh6_right + lh6_wrong)),
                  ((lh7_right) / (lh7_right + lh7_wrong)))
            print(step, "准确的估计准确率为",(((lh1_right) / (lh1_right + lh1_wrong))+((lh2_right) / (lh2_right + lh2_wrong))+
                  ((lh3_right) / (lh3_right + lh3_wrong))+((lh4_right) / (lh4_right + lh4_wrong))+
                  ((lh5_right) / (lh5_right + lh5_wrong))+((lh6_right) / (lh6_right + lh6_wrong))+
                  ((lh7_right) / (lh7_right + lh7_wrong)))/7)


            epoch_end_time = time()
            print("takes time:",(epoch_end_time-epoch_start_time), ' step:', step, ' accuracy:', acc," loss_fun:",loss_show)
            epoch_start_time = epoch_end_time
            # 偏差满足要求，保存模型
            if acc >= acc_rate:
                model_path = os.getcwd() + os.sep + '\models\\'+str(acc_rate) + "LSTM.model"
                saver.save(sess, model_path, global_step=step)
                break
            if step % 10 == 0 and step != 0:
                model_path = os.getcwd() + os.sep + '\models\\'  + str(acc_rate)+ "LSTM"+str(step)+".model"
                print(model_path)
                saver.save(sess, model_path, global_step=step)
            step += 1

    duration = time() - startTime
    print("total takes time:",duration)
    summary_writer.close()

    coord.request_stop()  # 通知线程关闭
    coord.join(threads)  # 等其他线程关闭这一函数才返回


