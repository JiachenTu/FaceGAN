#Collection of operations
import numpy as np
import tensorflow as tf

def init_weights(shape):
    fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    wscale = tf.constant(np.float32(std), name='wscale')
    return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    #return tf.Variable(tf.random_normal(shape=shape,stddev=0.02))

def init_bias(shape):
    return tf.Variable(tf.zeros(shape))

def conv2d(x,filter,strides,padding):
    return tf.nn.conv2d(x,filter, strides=strides, padding=padding)

def cost(labels,logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
