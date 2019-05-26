import numpy as np
import tensorflow as tf
from ops import *
from tensorflow.layers import batch_normalization
from tensorflow.keras.layers import UpSampling2D

class Generator:
    def __init__(self):
        self.layer_sizes = [512,256,128,3]
        with tf.variable_scope('g'):
            print("Initializing generator weights")
            # 100 = z input shape
            self.W1 = init_weights([100,7*7*self.layer_sizes[0]])
            self.W2 = init_weights([3,3,self.layer_sizes[0], self.layer_sizes[1]])
            self.W3 = init_weights([3,3,self.layer_sizes[1], self.layer_sizes[2]])
            self.W4 = init_weights([3,3,self.layer_sizes[2], self.layer_sizes[3]])

    def forward(self, X, momentum=0.5):
        z = tf.matmul(X,self.W1)
        z = tf.nn.leaky_relu(z)
        #Reshape to 4d tensor
        z = tf.reshape(z,[-1,7,7,self.layer_sizes[0]])

        #Upsampling to increase image size
        z = UpSampling2D()(z) #keras
        z = conv2d(z,self.W2,[1,1,1,1],padding="SAME")
        z = batch_normalization(z,momentum=momentum)
        z = tf.nn.leaky_relu(z)

        z = UpSampling2D()(z) #keras
        z = conv2d(z,self.W3,[1,1,1,1],padding="SAME")
        z = batch_normalization(z,momentum=momentum)
        z = tf.nn.leaky_relu(z)

        z = conv2d(z,self.W4,[1,1,1,1],padding="SAME")

        return tf.nn.tanh(z)
