import numpy as np
import tensorflow as tf
from ops import *
from tensorflow.layers import batch_normalization


from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten
from keras.layers.normalization import BatchNormalization



class Discriminator:
    def __init__(self, img_shape):
        _, _, channels = img_shape

        #initialize weights and bias
        ### Variable Scope to differentiate Generator and Discriminator
        layer_sizes = [64,64,128,256]
        with tf.variable_scope('d'):
            print("Initializing discriminator weights")
            self.W1 = init_weights([5,5,channels,layer_sizes[0]])
            self.b1 = init_bias([layer_sizes[0]])
            self.W2 = init_weights([3,3,layer_sizes[0],layer_sizes[1]])
            self.b2 = init_bias([layer_sizes[1]])
            self.W3 = init_weights([3,3,layer_sizes[1],layer_sizes[2]])
            self.b3 = init_bias([layer_sizes[2]])
            self.W4 = init_weights([2,2,layer_sizes[2],layer_sizes[3]])
            self.b4 = init_bias([layer_sizes[3]])
            self.W5 = init_weights([7*7*layer_sizes[3],1])
            self.b5 = init_bias([1])



    def forward(self, X, momentum=0.5):
        n_layers, use_sigmoid = 5, False
        ndf = 64
        x = Conv2D(filters=ndf, kernel_size=(4, 4), strides=2, padding='same')(X)
        x = LeakyReLU(0.2)(x)

        nf_mult, nf_mult_prev = 1, 1
        for n in range(n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
            x = Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)

        nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
        x = Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same')(x)
        if use_sigmoid:
            x = Activation('sigmoid')(x)

        x = Flatten()(x)
        x = Dense(1024, activation='tanh')(x)
        x = Dense(1, activation='sigmoid')(x)

        logits = tf.nn.bias_add(x, self.b5)
        return logits

    def forward_simple(self, X, momentum=0.5):
        # 1th layer
        z = conv2d(X,self.W1,[1,2,2,1],padding="SAME")  #Size 14,14,64
        #add bias
        z = tf.nn.bias_add(z,self.b1)
        #Activation Function
        z = tf.nn.leaky_relu(z)

        # 2nd layer
        z = conv2d(z,self.W2,[1,1,1,1],padding="SAME")  #Size 14,14,64
        z = tf.nn.bias_add(z,self.b2)
        z = batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        # 3th layer
        z = conv2d(z,self.W3,[1,2,2,1],padding="SAME")  #Size 7,7,128
        z = tf.nn.bias_add(z,self.b3)
        z = batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        # 4th layer
        z = conv2d(z,self.W4,[1,1,1,1],padding="SAME")  #Size 7,7,256
        z = tf.nn.bias_add(z,self.b4)
        z = batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        # Fully Connected Layer
        # Flatten Image
        z = tf.reshape(z,[-1, 7*7*256])
        logits = tf.matmul(z, self.W5)
        logits = tf.nn.bias_add(logits, self.b5)
        return logits   #Activation Function included in cost function
