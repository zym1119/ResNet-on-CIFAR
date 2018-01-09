# -*- coding: utf-8 -*-
# @Time    : 2018/1/8 20:35
# @Author  : Zhou YM
# @File    : resnet_main.py
# @Software: PyCharm
# @Project : resnet_cifar
# @Description:

import tensorflow as tf
import resnet_input
import resnet_model
import time


def train(hps, num_iterations, dataset):
    with tf.Graph().as_default():
        # input data
        images, labels = resnet_input.input(dataset, hps.batch_size, 'train')

        # resnet model
        model = resnet_model.ResNet(hps, images, labels, 'train')
        model.build_graph()

        # summary hook
        merged_summary_op = tf.summary.merge_all()

        # run session
        coord = tf.train.Coordinator()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config.gpu_options.allow_growth = True
        saver = tf.train.Saver()
        start_time = time.time()
        with tf.Session(config=config) as sess:
            # summaries
            train_writer = tf.summary.FileWriter('./train', graph=sess.graph)
            sess.run(tf.global_variables_initializer())
            queue_runner = tf.train.start_queue_runners(sess=sess, coord=coord)
            # train
            for i in range(num_iterations):
                # learning rate decay
                if i < 32000:
                    model.learning_rate = model.hps.init_lr
                elif i == 32000:
                    model.learning_rate /= 10
                elif i == 48000:
                    model.learning_rate /= 10

                _, acc, loss = sess.run([model.train_op, model.accuracy, model.cross_entropy])
                summary = sess.run(merged_summary_op)
                if i % 100 == 0:
                    train_writer.add_summary(summary, i)
                    print('iter %d, the loss is %.3f, accuracy on train set is %.2f' % (i, loss, acc))
                if i % 1000 == 0:
                    saver.save(sess, 'model/model.ckpt')
                    print('learning rate -> %f' % model.learning_rate)
            coord.request_stop()
            coord.join(queue_runner)
            train_writer.close()
            stop_time = time.time()
        print('%d iterations takes %.2f seconds' % (num_iterations, stop_time - start_time))


def evaluate(hps, num_iterations, dataset):
    total_acc = 0.0
    print('Loading trained network, please wait......')

    # input data
    images, labels = resnet_input.input(dataset, hps.batch_size, 'eval')

    # resnet model
    model = resnet_model.ResNet(hps, images, labels, 'eval')
    model.build_graph()

    # run session
    coord = tf.train.Coordinator()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        queue_runner = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver.restore(sess, './model/model.ckpt')
        for i in range(num_iterations):
            acc = sess.run(model.accuracy)
            total_acc += acc
        total_acc /= num_iterations
        print('Total accuracy on test set is %.2f' % total_acc)
        coord.request_stop()
        coord.join(queue_runner)


# DATASET & MODE #
# dataset = 'cifar10' or 'cifar100'
# mode = 'train' or 'eval'
dataset = 'cifar10'
mode = 'train'

if mode == 'train':
    num_iterations = 56000
    batch_size = 128
else:
    num_iterations = 100
    batch_size = 100

if dataset == 'cifar10':
    num_classes = 10
else:
    num_classes = 100

# set hyperparameters of resnet
hps = resnet_model.HParams(batch_size=batch_size,
                           num_classes=num_classes,
                           init_lr=0.1,
                           num_residual_units=5,
                           use_bottleneck=False,
                           weight_decay_rate=0.0001,
                           relu_leakiness=0.0,
                           optimizer='momentum')

if mode == 'train':
    train(hps, num_iterations, dataset)
else:
    evaluate(hps, num_iterations, dataset)
