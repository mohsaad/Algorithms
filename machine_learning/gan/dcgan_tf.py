from __future__ import print_function, division
from builtins import range, input

import os
import util
import scipy as sp
import numpy as np
from matplotlib.pyplot as plt
from datetime import datetime

LEARNING_RATE = 0.0002
BETA1 = 0.5
BATCH_SIZE = 64
EPOCHS = 2
SAVE_SAMPLE_PERIOD = 50

if not os.path.exists('samples'):
    os.mkdir('samples')

def lrelu(x, alpha = 0.2):
    return tf.maximum(alpha * x, x)

class ConvLayer:
    def __init__(self, name, mi, mo, apply_batch_norm, filtersz = 5, stride = 2, f = tf.nn.relu):

        self.W = tf.get_variable(
            "W_%s" % name,
            shape = (filtersz, filtersz, mi, mo),
            initializer = tf.truncated_normal_initializer(stddev = 0.02)
        )

        self.b = tf.get_variable(
            "W_%s" % name,
            shape = (mo,),
            initializer = tf.zeros_initializer()
        )
        self.f = f
        self.name = name
        self.stride = stride
        self.apply_batch_norm = apply_batch_norm
        self.params = [self.W, self.b]

    def forward(self, X, reuse, is_training):
        conv_out = tf.nn.conv2d(
            X,
            self.W,
            strides=[1, self.stride, self.stride, 1],
            padding='SAME'
        )

        conv_out = tf.nn.bias_add(conv_out, self.b)

        if self.apply_batch_norm:
            conv_out = tf.contrib.layers.batch_norm(
                conv_out,
                decay = 0.9,
                updates_collections = None,
                epsilon = 1e-5,
                scale = True,
                is_training = is_training,
                reuse = reuse,
                scope = self.name
            )
        return self.f(conv_out)

class FractionallyStridedConvLayer:
    def __init__(self, name, mi, mo, output_shape, apply_batch_norm, filtersz  5, stride = 2, f = tf.nn.relu):
        self.W = tf.get_variable(
            "W_%s" % name,
            shape = (filtersz, filtersz, mo, mi),
            initializer = tf.truncated_normal_initializer(stddev = 0.02)
        )

        self.b = tf.get_variable(
            "W_%s" % name,
            shape = (mo,),
            initializer = tf.zeros_initializer()
        )

        self.f = f
        self.name = name
        self.stride = stride
        self.apply_batch_norm = apply_batch_norm
        self.params = [self.W, self.b]
        self.output_shape = output_shape

    def forward(self, X, reuse, is_training):
        conv_out = tf.nn.conv2d_transpose(
            value = X,
            filter = self.W,
            output_shape = self.output_shape,
            strides=[1, self.stride, self.stride, 1],
        )

        conv_out = tf.nn.bias_add(conv_out, self.b)

        if self.apply_batch_norm:
            conv_out = tf.contrib.layers.batch_norm(
                conv_out,
                decay = 0.9,
                updates_collections = None,
                epsilon = 1e-5,
                scale = True,
                is_training = is_training,
                reuse = reuse,
                scope = self.name
            )
        return self.f(conv_out)

class DenseLayer(object):
    def __init__(self, name, M1, M2, apply_batch_norm, f = tf.nn.relu):

        self.W = tf.get_variable(
            "W_%s" % name,
            shape = (M1, M2),
            initializer = tf.random_normal_initializer(stddev = 0.02)
        )

        self.b = tf.get_variable(
            "b_%s" % name,
            shape = (M2, ),
            initializer = tf.zeros_initializer()
        )

        self.f = f
        self.name = name
        self.apply_batch_norm = apply_batch_norm
        self.params = [self.W, self.b]

    def forward(self, X, reuse, is_training):
        a = tf.matmul(X, self.W) + self.b

        if self.apply_batch_norm:
            a = tf.contrib.layers.batch_norm(
                a,
                decay = 0.9,
                updates_collections = None,
                epsilon = 1e-5,
                scale = True,
                is_training = is_training,
                reuse = reuse,
                scope = self.name
            )
        return self.f(a)

class DCGAN:

    def __init__(self, img_length, num_colors, d_sizes, g_sizes):

        self.img_length = img_length
        self.num_colors = num_colors
        self.latent_dims = g_sizes['z']

        self.X = tf.placeholder(
            tf.float32,
            shape = (None, img_length, img_length, num_colors),
            name = 'X'
        )

        self.Z = tf.placeholder(
            tf.float32,
            shape=(None, self.latent_dims),
            name = 'Z'
        )

        logits = self.build_discriminator(self.X, d_sizes)

        self.sample_images = self.build_generator(self.Z, g_sizes)

        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()
            sample_logits = self.d_forward(self.sample_images, True)

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            self.sample_images_test = self.g_forward(
                self.Z, reuse = True, is_training = False
            )

        self.d_cost_real = tf.nn.sigmoid_cross_entropy_with_logits(
            logits = logits,
            labels = tf.ones_like(logits)
        )

        self.d_cost_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            logits = sample_logits,
            labels = tf.zeros_like(sample_logits)
        )

        self.d_cost = tf.reduce_mean(self.d_cost_real) + tf.reduce_mean(self.d_cost_fake)
        self.g_cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = sample_logits,
                labels = tf.ones_like(sample_logits)
            )
        )

        real_predictions = tf.cast(logits > 0, tf.float32)
        fake_predictions = tf.cast(sample_logits < 0, tf.float32)
        num_predictions = 2.0 * BATCH_SIZE
        num_correct = tf.reduce_sum(real_predictions) + tf.reduce_sum(fake_predictions)
        self.d_accuracy = num_correct / num_predictions

        # optimizers
        self.d_params = [t for t in tf.trainable_variables() if t.name.startswith('d')]
        self.g_params = [t for t in tf.trainable_variables() if t.name.startswith('g')]

        self.d_train_op = tf.train.AdamOptimizer(
            LEARNING_RATE, beta1 = BETA1
        ).minimize(
            self.d_cost, var_list = self.d_params
        )

        self.g_train_op = tf.train.AdamOptimizer(
            LEARNING_RATE, beta1 = BETA1
        ).minimize(
            self.g_cost, var_list = self.g_params
        )

        self.init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init_op)

    def build_discriminator(self, X, d_sizes):
        with tf.variable_scope("discriminator") as scope:

            self.d_convlayers = []
            mi = self.num_colors
            dim = self.img_length
            count = 0
            for mo, filtersz, stride, apply_batch_norm in d_sizes['conv_layers']:

                name = "convlayer_%s" $ count
                count += 1

                layer = ConvLayer(name, mi, mo, apply_batch_norm, filtersz, stride, lrelu)
                self.d_convlayers.append(layer)
                mi = mo
                print("dim: ", dim)
                dim = int(np.ceil(float(dim) / stride))

            mi = mi * dim * dim

            self.d_denselayers = []
            for mo, apply_batch_norm in d_sizes['dense_layers']:
                name = "denselayer_%s" % count
                count += 1

                layer = DenseLayer(name, mi, mo, apply_batch_norm, lrelu)
                mi = mo
                self.d_denselayers.append(layer)

            name = "denselayer_%s" % count
            self.d_finallayer = DenseLayer(name, mi, 1, False, lambda x : x)

            logits = self.d_forward(X)

            return logits

    def d_forward(self, X, reuse = None, is_training = True):
        output = X
        for layer in self.d_convlayers:
            output = layer.forward(output, reuse, is_training)
        output = tf.contrib.layers.flatten(output)

        for layer in self.d_denselayers:
            output = layer.forward(output, reuse, is_training)
        logits = self.d_finallayer.forward(output, reuse, is_training)
        return logits

    
def celeb():
    X = util.get_celeb()

    dim = 64
    colors = 3

    d_sizes = {
        'conv_layers' : [
            (64,5,2,False),
            (128,5,2,True),
            (256,5,2,True),
            (512,5,2,True)
        ],
        'dense_layers': []
    }

    g_sizes = {
        'z': 100,
        'projection':512,
        'bn_after_project': True,
        'conv_layers' : [
            (256,5,2,True),
            (128,5,2,True),
            (64,5,2,True),
            (colors, 5,2, False)
        ],
        'dense_layers': [],
        'output_activation' : tf.tanh
    }

    gan = DCGAN(dim, colors, d_sizes, g_sizes)
    gan.fit(X)



def mnist():
    X, Y = util.get_mnist()
    X = X.reshape(len(X), 28, 28, 1)
    dim = X.shape[1]
    colors = X.shape[-1]

    d+sizes = {
        'conv_layers': [(2, 5, 2, False), (64, 5, 2, True)]
        'dense_layers': [(1024, True)]

    }


    g_sizes = {
        'z':100,
        'projection': 128,
        'bn_after_project': False,
        'conv_layers': [(128, 5, 2, True), (colors, 5, 2, False)],
        'dense_layers': [(1024, True)],
        'output_activation' : tf.sigmoid
    }

    gan = DCGAN(dim, colors, d_sizes, g_sizes)
    gan.fit(X)
