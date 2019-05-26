import numpy as np
import tensorflow as tf
from ops import *
from tensorflow.layers import batch_normalization

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
