#This code is from https://github.com/grausof/keras-sincnet

import tensorflow as tf
import numpy as np
import math
import os
from keras.models import Model
from keras.layers import (
    Input, Activation, Add, BatchNormalization, LeakyReLU, AveragePooling1D, 
    concatenate, Lambda, UpSampling1D, Subtract, Reshape)
from keras.optimizers import Adam
from keras.layers.convolutional import Conv1D
from keras import backend as K
import pdb

class Sinc_Conv_Layer():
    def __init__(self, input_size, N_filt, Filt_dim, fs, NAME):
        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10**(mel_points / 2595) - 1)) # Convert Mel to Hz
        b1=np.roll(f_cos,1).astype(np.float32)
        b2=np.roll(f_cos,-1).astype(np.float32)
        b1[0]=30
        b2[-1]=(fs/2)-100
                
        self.freq_scale=fs*1.0
        self.NAME = NAME
        band = tf.constant((b2-b1)/self.freq_scale)
        b1 = tf.constant(b1/self.freq_scale)
        self.filt_b1 = tf.get_variable(self.NAME+"_b1", initializer=b1)
        self.filt_band = tf.get_variable(self.NAME+"_band", initializer=band)
        self.N_filt=N_filt
        self.Filt_dim=Filt_dim
        self.fs=fs
        self.input_size = input_size

    def input_channel_slice(self, x, slice_index):
        return tf.expand_dims(x[:, :, slice_index],axis=-1)

    def compute_output(self, input_tensor):
        
        filted = Lambda(self.sinc_conv)(input_tensor)
        return filted

    def sinc_conv(self, input_tensor):
        N = self.Filt_dim
        t_right = tf.linspace(1.0, (N-1)/2, int((N-1)/2)) / self.fs

        min_freq = 50.0
        min_band = 50.0
        
        filt_beg_freq=tf.abs(self.filt_b1)+min_freq/self.freq_scale
        filt_end_freq=filt_beg_freq+(tf.abs(self.filt_band)+min_band/self.freq_scale)

        n=tf.linspace(0.0, N, N)

        # Filter window (hamming)
        window = 0.54-0.46*tf.cos(2*math.pi*n/N)
        filter_tmp = []
        for i in range(self.N_filt):
                        
            low_pass1 = 2*filt_beg_freq[i] * self.sinc(t_right*(2*math.pi*filt_beg_freq[i]*self.freq_scale))
            low_pass2 = 2*filt_end_freq[i] * self.sinc(t_right*(2*math.pi*filt_end_freq[i]*self.freq_scale))
            band_pass=(low_pass2-low_pass1)

            band_pass=band_pass/tf.reduce_max(band_pass)
            filter_tmp.append(band_pass*window)

        filters = tf.reshape(tf.stack(filter_tmp, axis=1),(self.Filt_dim, 1, self.N_filt), name=self.NAME+"_filters")
        out=tf.nn.conv1d(input_tensor, filters, stride=256, padding='SAME')
        return out

    def sinc(self, x):
        atzero = tf.divide(tf.sin(x),1)
        atother = tf.divide(tf.sin(x),x)
        value = tf.where(tf.equal(x,0), atzero, atother )
        return tf.concat([tf.reverse(value, axis=[0]), tf.constant(1,dtype=tf.float32, shape=[1,]), value], axis=0)
