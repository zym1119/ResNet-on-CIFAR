# -*- coding: utf-8 -*-
# @Time    : 2018/1/8 14:33
# @Author  : Zhou YM
# @File    : resnet_model.py
# @Software: PyCharm
# @Project : resnet_cifar
# @Description:

"""
    ResNet Model
"""
from collections import namedtuple
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages


HParams = namedtuple('HParams',
                     ['batch_size', 'num_classes', 'init_lr', 'num_residual_units', 'use_bottleneck',
                      'weight_decay_rate', 'relu_leakiness', 'optimizer'])


class ResNet(object):

    def __init__(self, hps, images, labels, mode):
        """
        ResNet constructor
        :param hps: hyperparameters
        :param images: batch of images, 4D tensor of [batch_size, image_size, image_size, depth]
        :param labels: batch of labels, 1D tensor of [batch_size]
        :param mode: 'train' or 'test'
        """
        self.hps = hps
        self.images = images
        self.labels = labels
        self.mode = mode
        self.learning_rate = hps.init_lr
        self._moving_averages = []

    """####################"""
    """   Building Model   """
    """####################"""

    def build_graph(self):
        self.global_step = tf.Variable(0.0, trainable=False, name='global_step')

        self._build_model()

        if self.mode == 'train':
            self._build_train_op()

        self.summaries = tf.summary.merge_all()

    def _build_model(self):
        with tf.variable_scope('init'):
            x = self.images
            # first convolution
            # x = self._conv('init_conv', x, 3, 3, 64, [1, 1, 1, 1])
            x = self._conv('init_conv', x, 3, 3, 16, [1, 1, 1, 1])

        # residual network parameters
        # if self.hps.use_bottleneck:
        #     # res_func =
        #     pass
        # else:
        res_func = self._residual
        # standard resnet channel
        # channels = [64, 128, 256, 512]
        # cifar resnet channel
        channels = [16, 32, 64]
        num_res_units = self.hps.num_residual_units

        # Residual Group 1
        for i in range(num_res_units):
            with tf.variable_scope('res_unit_1_%d' % i):
                x = res_func(x, channels[0], channels[0], [1, 1, 1, 1], False)

        # Residual Group 2
        with tf.variable_scope('res_unit_2_0'):
            x = res_func(x, channels[0], channels[1], [1, 2, 2, 1], False)
        for i in range(1, num_res_units):
            with tf.variable_scope('res_unit_2_%d' % i):
                x = res_func(x, channels[1], channels[1], [1, 1, 1, 1], False)

        # Residual Group 3
        with tf.variable_scope('res_unit_3_0'):
            x = res_func(x, channels[1], channels[2], [1, 2, 2, 1], False)
        for i in range(1, num_res_units):
            with tf.variable_scope('res_unit_3_%d' % i):
                x = res_func(x, channels[2], channels[2], [1, 1, 1, 1], False)

        # Global average pooling
        with tf.variable_scope('global_avg_pool'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._global_average_pooling(x)

        # Fully connected layer with softmax
        with tf.variable_scope('fc_softmax'):
            self.logits = self._fully_connected(x, self.hps.num_classes)

        # Compute loss
        with tf.variable_scope('costs'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels,
                                                                           name='cross_entropy_per_example')
            self.cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy')
            self.cross_entropy += self._weight_decay()
            tf.summary.scalar('cost', self.cross_entropy)

        # Compute accuracy
        predictions = tf.cast(tf.nn.in_top_k(self.logits, self.labels, 1), tf.float32)
        self.accuracy = tf.reduce_mean(predictions)
        tf.summary.scalar('accuracy', self.accuracy)

    def _build_train_op(self):
        # learning rate
        # self._set_lr(self.hps.init_lr, self.global_step)
        tf.summary.scalar('learning_rate', self.learning_rate)

        # compute gradient
        if self.hps.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        elif self.hps.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        apply_gradient_op = optimizer.minimize(self.cross_entropy, self.global_step)

        self.train_op = tf.group(*(self._moving_averages+[apply_gradient_op]))

    # residual unit
    def _residual(self, x, in_channel, out_channel, stride, activation_before_residual=False):
        orig_x = x
        if activation_before_residual:
            # original residual unit
            # |
            # |\
            # | \
            # |  | Conv
            # |  | BN
            # |  | ReLU
            # |  | Conv
            # |  | BN
            # | /
            # |/
            # + Add
            # | ReLU
            # x
            with tf.variable_scope('shared_activation') as scope:
                x = self._conv('conv1', x, 3, in_channel, out_channel, stride)
                x = self._batch_norm('bn1', x)
                x = self._relu(x, self.hps.relu_leakiness)
                x = self._conv('conv2', x, 3, out_channel, out_channel, [1, 1, 1, 1])
                x = self._batch_norm('bn2', x)
                x = self._add(x, orig_x, in_channel, out_channel, scope)
                x = self._relu(x, self.hps.relu_leakiness)
        else:
            # pre-activation residual unit
            # |
            # |\
            # | \
            # |  | BN
            # |  | ReLU
            # |  | Conv
            # |  | BN
            # |  | ReLU
            # |  | Conv
            # | /
            # |/
            # + Add
            # |
            # x
            with tf.variable_scope('residual_only_activation') as scope:
                x = self._batch_norm('bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                x = self._conv('conv1', x, 3, in_channel, out_channel, stride)
                x = self._batch_norm('bn1', x)
                x = self._relu(x, self.hps.relu_leakiness)
                x = self._conv('conv2', x, 3, out_channel, out_channel, [1, 1, 1, 1])
                x = self._add(x, orig_x, in_channel, out_channel, scope)
        return x

    def _fully_connected(self, x, num_classes):
        x = tf.reshape(x, [self.hps.batch_size, -1])
        weights = tf.get_variable('weights', [x.get_shape()[1], num_classes], tf.float32,
                                  initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        bias = tf.get_variable('bias', [num_classes], tf.float32, tf.constant_initializer())

        return tf.nn.xw_plus_b(x, weights, bias)

    """##########################"""
    """   Computation Function   """
    """##########################"""

    # global average pooling
    def _global_average_pooling(self, x):
        # feature map size 8*8
        return tf.reduce_mean(x, [1, 2])

    # addition in residual unit
    def _add(self, x, orig_x, in_channel, out_channel, scope):
        with tf.variable_scope(scope):
            if in_channel != out_channel:
                orig_x = self._conv('shortcut_conv', orig_x, 1, in_channel, out_channel, [1, 2, 2, 1])
            y = x+orig_x
        return y

    # 2D convolution
    def _conv(self, name, x, conv_size, in_channel, out_channel, stride):
        with tf.variable_scope(name):
            kernel = tf.get_variable('weights', [conv_size, conv_size, in_channel, out_channel], tf.float32,
                                     tf.contrib.layers.xavier_initializer_conv2d(tf.float32))
            conv_out = tf.nn.conv2d(x, kernel, stride, padding='SAME')
        return  conv_out

    # leaky relu
    def _relu(self, x, leakiness=0.0):
        # x>=0 return x, x<0 return leakiness * x
        return tf.where(tf.greater_equal(x, 0.0), x, leakiness*x, name='leaky_relu')

    # batch normalization
    def _batch_norm(self, name, x):
        with tf.variable_scope(name):
            # parameters
            # beta&gamma is trainable, learned from dataset; mean&variance are moving average of dataset
            shape = [x.get_shape()[-1]]
            beta = tf.get_variable('beta', shape, tf.float32, tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', shape, tf.float32, tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
                moving_mean = tf.get_variable('moving_mean', shape, tf.float32,
                                              initializer=tf.constant_initializer(0.0, tf.float32),
                                              trainable=False)
                moving_variance = tf.get_variable('moving_variance', shape, tf.float32,
                                              initializer=tf.constant_initializer(1.0, tf.float32),
                                              trainable=False)
                self._moving_averages.append(moving_averages.assign_moving_average(moving_mean, mean, 0.9))
                self._moving_averages.append(moving_averages.assign_moving_average(moving_variance,
                                                                                   variance, 0.9))
            else:
                # 对于get_variable()来说，如果已经创建的变量对象，就把那个对象返回，如果没有创建变量对象的话，就创建一个新的。
                mean = tf.get_variable('moving_mean', shape, tf.float32,
                                       initializer=tf.constant_initializer(0.0),
                                       trainable=False)
                variance = tf.get_variable('moving_variance',
                                           shape, tf.float32,
                                           initializer=tf.constant_initializer(1.0, tf.float32),
                                           trainable=False)

                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)

            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-5)
            y.set_shape(x.get_shape())

            return y

    def _weight_decay(self):
        cost = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'weights') > 0:
                cost.append(tf.nn.l2_loss(var))
        return tf.multiply(tf.add_n(cost), self.hps.weight_decay_rate)

    def _set_lr(self, init_lr, global_step):
        if global_step < 32000:
            self.learning_rate = init_lr
        elif global_step == 32000:
            self.learning_rate /= 10
        elif global_step == 48000:
            self.learning_rate /= 10
        else:
            self.learning_rate = 0.0001
