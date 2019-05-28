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
        latents_in = X                          # First input: Latent vectors [minibatch, latent_size].
        labels_in           = tf.placeholder(tf.float32, [None,0])         # Second input: Labels [minibatch, label_size].
        num_channels        = 3            # Number of output color channels. Overridden based on dataset.
        resolution          = 32           # Output resolution. Overridden based on dataset.
        label_size          = 0            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        fmap_base           = 8192         # Overall multiplier for the number of feature maps.
        fmap_decay          = 1.0          # log2 feature map reduction when doubling the resolution.
        fmap_max            = 512          # Maximum number of feature maps in any layer.
        latent_size         = None         # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
        normalize_latents   = True         # Normalize latent vectors before feeding them to the network?
        use_wscale          = True         # Enable equalized learning rate?
        use_pixelnorm       = False        # Enable pixelwise feature vector normalization?
        pixelnorm_epsilon   = 1e-8         # Constant epsilon for pixelwise feature vector normalization.
        use_leakyrelu       = True         # True = leaky ReLU, False = ReLU.
        dtype               = 'float32'    # Data type to use for activations and outputs.
        fused_scale         = True         # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
        structure           = None         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
        is_template_graph   = False        # True = template graph constructed by the Network class, False = actual evaluation.

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4
        def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
        if latent_size is None: latent_size = nf(0)
        if structure is None: structure = 'linear' if is_template_graph else 'recursive'
        act = leaky_relu if use_leakyrelu else tf.nn.relu

        latents_in.set_shape([None, latent_size])
        labels_in.set_shape([None, label_size])
        combo_in = tf.cast(tf.concat([latents_in, labels_in], axis=1), dtype)
        lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

        # Building blocks.
        def block(x, res): # res = 2..resolution_log2
            with tf.variable_scope('%dx%d' % (2**res, 2**res)):
                if res == 2: # 4x4
                    if normalize_latents: x = pixel_norm(x, epsilon=pixelnorm_epsilon)
                    with tf.variable_scope('Dense'):
                        x = dense(x, fmaps=nf(res-1)*16, gain=np.sqrt(2)/4, use_wscale=use_wscale) # override gain to match the original Theano implementation
                        x = tf.reshape(x, [-1, nf(res-1), 4, 4])
                        x = PN(act(apply_bias(x)))
                    with tf.variable_scope('Conv'):
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                else: # 8x8 and up
                    if fused_scale:
                        with tf.variable_scope('Conv0_up'):
                            x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                    else:
                        x = upscale2d(x)
                        with tf.variable_scope('Conv0'):
                            x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                    with tf.variable_scope('Conv1'):
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                return x
        def torgb(x, res): # res = 2..resolution_log2
            lod = resolution_log2 - res
            with tf.variable_scope('ToRGB_lod%d' % lod):
                return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))

        # Linear structure: simple but inefficient.
        if structure == 'linear':
            x = block(combo_in, 2)
            images_out = torgb(x, 2)
            for res in range(3, resolution_log2 + 1):
                lod = resolution_log2 - res
                x = block(x, res)
                img = torgb(x, res)
                images_out = upscale2d(images_out)
                with tf.variable_scope('Grow_lod%d' % lod):
                    images_out = lerp_clip(img, images_out, lod_in - lod)

        # Recursive structure: complex but efficient.
        if structure == 'recursive':
            def grow(x, res, lod):
                y = block(x, res)
                img = lambda: upscale2d(torgb(y, res), 2**lod)
                if res > 2: img = cset(img, (lod_in > lod), lambda: upscale2d(lerp(torgb(y, res), upscale2d(torgb(x, res - 1)), lod_in - lod), 2**lod))
                if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
                return img()
            images_out = grow(combo_in, 2, resolution_log2 - 2)

        assert images_out.dtype == tf.as_dtype(dtype)
        z = tf.identity(images_out, name='images_out')
        return tf.nn.tanh(z)




    def forward_simple(self, X, momentum=0.5):
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
