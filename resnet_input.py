# -*- coding: utf-8 -*-
# @Time    : 2018/1/8 10:54
# @Author  : Zhou YM
# @File    : resnet_input.py
# @Software: PyCharm
# @Project : resnet_cifar
# @Description:

"""
    CIFAR dataset input
"""

import os
import tensorflow as tf


def input(data_set, batch_size, mode):
    """
    Build CIFAR dataset inputs and labels
    :param data_set: 'cifar10' or 'cifar100'
    :param batch_size: size per batch
    :param mode: 'train' or 'eval'
    :return:
        images: batch of images, 4D tensor of [batch_size, image_size, image_size, 3]
        labels: batch of labels, 2D tensor of [batch_size]
    """
    image_size = 32
    if data_set == 'cifar10':
        label_offset = 0
        num_classes = 10
        filepath = os.path.join('E:\Dataset', 'cifar-10\cifar-10-batches-bin')
    elif data_set == 'cifar100':
        label_offset = 1
        num_classes = 100
        filepath = os.path.join('E:\Dataset', 'cifar-100-binary')
    else:
        raise ValueError('Cannot find dataset %s' % data_set)
    label_bytes = 1
    depth = 3
    image_bytes = image_size*image_size*depth
    record_bytes = image_bytes+label_offset+label_bytes

    # file reader
    if mode == 'train':
        if data_set == 'cifar10':
            filenames = [os.path.join(filepath, 'data_batch_%d.bin' % i) for i in range(1, 6)]
        else:
            filenames = [os.path.join(filepath, 'train.bin')]
    else:
        if data_set == 'cifar10':
            filenames = [os.path.join(filepath, 'test_batch.bin')]
        else:
            filenames = [os.path.join(filepath, 'test.bin')]
    file_queue = tf.train.string_input_producer(filenames)
    reader = tf.FixedLengthRecordReader(record_bytes)
    _, value = reader.read(file_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)

    # slice label and image
    # coarse_labels -> 0-19, fine_labels -> 0-99(what we need)
    label = tf.cast(tf.slice(record_bytes, begin=[label_offset], size=[label_bytes]), tf.int32)
    image = tf.slice(record_bytes, begin=[label_bytes+label_offset], size=[image_bytes])
    image_reshaped = tf.reshape(image, [depth, image_size, image_size])
    image = tf.cast(tf.transpose(image_reshaped, [1, 2, 0]), tf.float32)

    # data augmentation
    if mode == 'train':
        image = tf.image.resize_image_with_crop_or_pad(image, image_size + 4, image_size + 4)
        image = tf.random_crop(image, [image_size, image_size, depth])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.per_image_standardization(image)
    else:
        # image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        image = tf.image.per_image_standardization(image)

    # set shape
    image.set_shape([image_size, image_size, 3])
    label.set_shape([1])

    # set queue
    min_queue_examples = int(80*batch_size)
    capacity = int(min_queue_examples+10*batch_size)
    if mode == 'train':
        images, labels = tf.train.shuffle_batch([image, label], batch_size, capacity, min_queue_examples,
                                                num_threads=4)
        tf.summary.image('images', images, 10)
    else:
        images, labels = tf.train.batch([image, label], batch_size, num_threads=4, capacity=10000)
        tf.summary.image('test_images', images, 10)

    labels = tf.reshape(labels, [batch_size])

    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert len(labels.get_shape()) == 1
    assert labels.get_shape()[0] == batch_size

    return images, labels

