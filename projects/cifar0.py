import os
import scipy.misc

import tensorflow as tf
from dependency import cifar10
from dependency import cifar10_train
from dependency import cifar10_input

os.chdir("/Users/shifengchen/Development/github/learningDL")


def inputs_origin(data_dir):
    filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError("Failed to find file: " + f)
    filename_queue = tf.train.string_input_producer(filenames)
    read_input = cifar10_input.read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    return reshaped_image


if __name__ == '__main__':

    # with tf.Session() as sess:
    #     reshaped_image = inputs_origin("data/cifar/cifar-10-batches-bin")
    #     threads = tf.train.start_queue_runners(sess=sess)
    #     sess.run(tf.global_variables_initializer())
    #     raw_path = "data/cifar/raw/"
    #     if not os.path.exists(raw_path):
    #         os.mkdir(raw_path)
    #     for i in range(30):
    #         image_array = sess.run(reshaped_image)
    #         scipy.misc.toimage(image_array).save(raw_path + "%d.jpg" % i)

    FLAGS = tf.app.flags.FLAGS
    FLAGS.data_dir = "data/cifar/"
    FLAGS.train_dir = "data/cifar/train/"

    cifar10_train.main()


