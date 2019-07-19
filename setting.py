'''
配置文件，定义超参数
'''
import tensorflow as tf
# 每次使用100条数据进行训练
batch_size = 16
# 图像向量
width = 48
height = 48
# LSTM
rnn_size = 256
# 输出层one-hot向量长度的
out_size = 7
# 通道数
channel = 3
# 训练次数
num_epoch = 10
# 学习率
learning_rate = 0.001
# learning_rate = tf.train.exponential_decay(learning_rate=0.5, global_step=num_epoch, decay_steps=10, decay_rate=0.9, staircase=False)
