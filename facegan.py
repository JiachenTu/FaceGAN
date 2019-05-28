import numpy as np
import tensorflow as tf
from ops import *
import matplotlib.pyplot as plt
import os
from generator import Generator
from discriminator import Discriminator
from keras.preprocessing import image
import cv2


class DCGAN:
    def __init__(self, img_shape, sample_folder_name, iterations=5000, lr_gen=0.0001, lr_dc=0.00005, z_shape=100, batch_size=64 , beta1=0.7, sample_interval=1000):

        #Create sample folder
        if not os.path.exists(f"{sample_folder_name}/"):
            os.makedirs(f"{sample_folder_name}/")

        self.SAMPLE_FOLDER_NAME = sample_folder_name
        #Unpack Image shape
        self.rows, self.cols, self.channels = img_shape
        self.batch_size=batch_size
        self.iterations = iterations
        self.z_shape = z_shape
        self.sample_interval = sample_interval
        self.generator = Generator()
        self.discriminator = Discriminator(img_shape)

        #Load CelebA dataset
        dir_data      = "./data/celebA/"
        Ntrain        = 200000
        Ntest         = 100
        nm_imgs       = np.sort(os.listdir(dir_data))
        ## name of the jpg files for training set
        nm_imgs_train = nm_imgs[:Ntrain]
        ## name of the jpg files for the testing data
        nm_imgs_test  = nm_imgs[Ntrain:Ntrain + Ntest]
        img_shape     = (32, 32, 3)


        X_train = []
        for i, myid in enumerate(nm_imgs_train):
            im = image.load_img(dir_data + "/" + myid,
                         target_size=img_shape[:2])
            im = image.img_to_array(im)
            X_train.append(im)
        X = np.array(X_train)

        #Values 0~255
        #Scale -1~1
        self.X = X/127.5 - 1


        #Create placeholders for input
        self.phX = tf.placeholder(tf.float32, [None, self.rows, self.cols, self.channels])
        self.phZ = tf.placeholder(tf.float32, [None,self.z_shape])

        #Generate forward pass
        self.gen_out = self.generator.forward(self.phZ)

        #Discriminator predictions
        #Fake IMG
        dc_logits_fake = self.discriminator.forward(self.gen_out)
        #Real IMG
        dc_logits_real = self.discriminator.forward(self.phX)

        #cost functions
        #fake -- 0; real -- 1
        dc_fake_loss = cost(tf.zeros_like(dc_logits_fake),dc_logits_fake)
        dc_real_loss = cost(tf.ones_like(dc_logits_real),dc_logits_real)

        self.dc_loss = tf.add(dc_fake_loss,dc_real_loss)
        #Generator tries to fool D so that it outputs 1 for fake IMGs
        self.gen_loss = cost(tf.ones_like(dc_logits_fake),dc_logits_fake)

        #Collect trainable variables
        train_vars = tf.trainable_variables()

        #Differentiate G and D variables
        dc_vars = [var for var in train_vars if 'd' in var.name]
        gen_vars = [var for var in train_vars if 'g' in var.name]

        #Create training variables
        self.dc_train = tf.train.AdamOptimizer(lr_dc, beta1=beta1).minimize(self.dc_loss,var_list=dc_vars)
        self.gen_train = tf.train.AdamOptimizer(lr_gen, beta1=beta1).minimize(self.gen_loss,var_list=gen_vars)


    def train(self):
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        #Init all vars
        self.sess.run(init)

        #Start training loop
        for i in range(self.iterations):
            #rand batch and indices
            idx = np.random.randint(0,len(self.X),self.batch_size)
            batch_X = self.X[idx]
            batch_Z = np.random.uniform(-1,1,(self.batch_size,self.z_shape))

            #Train D and store dc loss
            batch_X = batch_X.reshape([-1,32,32,self.channels])
            _, d_loss = self.sess.run([self.dc_train,self.dc_loss],feed_dict={self.phX:batch_X,self.phZ:batch_Z})

            #Create new batch for G
            batch_Z = np.random.uniform(-1,1,(self.batch_size,self.z_shape))

            #Train G and store G loss
            _, g_loss = self.sess.run([self.gen_train,self.gen_loss], feed_dict={self.phZ:batch_Z})


            #Generate samples and print loss
            if i % self.sample_interval == 0:
                self.generate_sample(i)
                print(f"Epoch:{i}. Discriminator loss: {d_loss}. Generator loss {g_loss}")


    def generate_sample(self, iteration):
        # 5 samples per IMG
        c,r = 5, 5


        # New input for sample, 5*5 = 25 IMGs
        z = np.random.uniform(-1,1,(25,self.z_shape))
        imgs = self.sess.run(self.gen_out, feed_dict={self.phZ:z})


        #Scale back to values (0,1), currently (-1,1)
        imgs = imgs*0.5+0.5

        #imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

        fig,axs = plt.subplots(c,r)
        count = 0
        for i in range(c):
            for j in range(r):
                axs[i,j].imshow(imgs[count,:,:,0])
                axs[i,j].axis('off')
                count += 1

        # save image
        fig.savefig(f"{self.SAMPLE_FOLDER_NAME}/{iteration}.png")
        plt.close()




if __name__ == '__main__':
    img_shape = (32,32,3)
    dcgan = DCGAN(img_shape,"samples")
    dcgan.train()
