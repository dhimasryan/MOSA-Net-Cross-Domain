"""
@author: Ryandhimas Zezario
ryandhimas@citi.sinica.edu.tw
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import keras
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import math
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Sequential, model_from_json, Model, load_model
from keras.layers import Layer, concatenate
from keras.layers.core import Dense, Dropout, Flatten, Activation, Reshape, Lambda
from keras.activations import softmax
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.backend import squeeze
from keras.layers import LSTM, TimeDistributed, Bidirectional, dot, Input, CuDNNLSTM
from keras.constraints import max_norm
from keras_self_attention import SeqSelfAttention
from SincNet import Sinc_Conv_Layer
import argparse
import tensorflow as tf
import scipy.io
import scipy.stats
import librosa
import time  
import numpy as np
import numpy.matlib
import random
import pdb
random.seed(999)

epoch=50
batch_size=1
forgetgate_bias=-3

NUM_EandN=15000
NUM_Clean=1500

def attention_model() :
    input_size =(None,1)
    _input = Input(shape=(None, 257))
    _input_end2end = Input(shape=(None, 1))

    SincNet_ = Sinc_Conv_Layer(input_size, N_filt = 257, Filt_dim = 251, fs = 16000, NAME = "SincNet").compute_output(_input_end2end)
    merge_input = concatenate([_input, SincNet_],axis=1) 
    re_input = keras.layers.core.Reshape((-1, 257, 1), input_shape=(-1, 257))(merge_input)
        
    # CNN
    conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(re_input)
    conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
    conv1 = (Conv2D(16, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv1)
        
    conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
    conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
    conv2 = (Conv2D(32, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv2)
        
    conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
    conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
    conv3 = (Conv2D(64, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv3)
        
    conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
    conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv4)
    conv4 = (Conv2D(128, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv4)
        
    re_shape = keras.layers.core.Reshape((-1, 4*128), input_shape=(-1, 4, 128))(conv4)
    _input_hubert = Input(shape=(None, 1024))
    bottleneck=TimeDistributed(Dense(512, activation='relu'))(_input_hubert)
    concat_with_wave2vec = concatenate([re_shape, bottleneck],axis=1) 
    blstm=Bidirectional(CuDNNLSTM(128, return_sequences=True), merge_mode='concat')(concat_with_wave2vec)

    flatten = TimeDistributed(keras.layers.core.Flatten())(blstm)
    dense1=TimeDistributed(Dense(128, activation='relu'))(flatten)
    dense1=Dropout(0.3)(dense1)

    attention = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,kernel_regularizer=keras.regularizers.l2(1e-4),bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4, name='Attention')(dense1)
    Frame_score=TimeDistributed(Dense(1), name='Frame_score')(attention)
    PESQ_score=GlobalAveragePooling1D(name='PESQ_score')(Frame_score)

    attention2 = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,kernel_regularizer=keras.regularizers.l2(1e-4),bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4, name='Attention2')(dense1)
    Frame_stoi=TimeDistributed(Dense(1, activation='sigmoid'), name='Frame_stoi')(attention2)
    STOI_score= GlobalAveragePooling1D(name='STOI_score')(Frame_stoi)

    attention3 = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,kernel_regularizer=keras.regularizers.l2(1e-4),bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4, name='Attention3')(dense1)
    Frame_sdi=TimeDistributed(Dense(1), name='Frame_sdi')(attention3)
    SDI_score= GlobalAveragePooling1D(name='SDI_score')(Frame_sdi)

    model = Model(outputs=[PESQ_score, Frame_score, STOI_score , Frame_stoi, SDI_score, Frame_sdi], inputs=[_input,_input_end2end, _input_hubert])
    return model	
       
def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path

def Sp_and_phase(path, Noisy=False):
    
    signal, rate  = librosa.load(path,sr=16000)
    signal=signal/np.max(abs(signal)) 
    F = librosa.stft(signal,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
      
    Lp=np.abs(F)
    phase=np.angle(F)
    if Noisy==True:    
        meanR = np.mean(Lp, axis=1).reshape((257,1))
        stdR = np.std(Lp, axis=1).reshape((257,1))+1e-12
        NLp = (Lp-meanR)/stdR
    else:
        NLp=Lp
    
    NLp=np.reshape(NLp.T,(1,NLp.shape[1],257))
    end2end = np.reshape(signal,(1,signal.shape[0],1))
    return NLp, end2end

def data_generator(file_list, file_list_whisper):
	index=0
	while True:
         wer_filepath=file_list[index].split(',')                
         noisy_LP,noisy_end2end =Sp_and_phase(wer_filepath[2]) 
         
         noisy_whisper =np.load(file_list_whisper[index]) 
         noisy_whisper =np.reshape(noisy_whisper,(1,noisy_whisper.shape[0],1024))

         intell=np.asarray(float(wer_filepath[1])).reshape([1])
         qual=np.asarray(float(wer_filepath[0])).reshape([1])         

         feat_length_end2end = math.ceil(float(noisy_end2end.shape[1])/256)
         final_len = noisy_LP.shape[1] + int(feat_length_end2end) + noisy_whisper.shape[1]

         index += 1
         if index == len(file_list):
             index = 0
            
             random.Random(7).shuffle(file_list)
             random.Random(7).shuffle(file_list_whisper)

         yield  [noisy_LP, noisy_end2end, noisy_whisper], [qual, qual[0]*np.ones([1,final_len,1]),intell, intell[0]*np.ones([1,final_len,1])]

def denorm(input_x):
    input_x = input_x*(5-0) + 0
    return input_x
    
def BLSTM_CNN_with_ATT_cross_domain():
    input_size =(None,1)
    _input = Input(shape=(None, 257))
    _input_end2end = Input(shape=(None, 1))

    SincNet_ = Sinc_Conv_Layer(input_size, N_filt = 257, Filt_dim = 251, fs = 16000, NAME = "SincNet_1").compute_output(_input_end2end)
    merge_input = concatenate([_input, SincNet_],axis=1) 
    re_input = keras.layers.core.Reshape((-1, 257, 1), input_shape=(-1, 257))(merge_input)
        
    # CNN
    conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(re_input)
    conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
    conv1 = (Conv2D(16, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv1)
        
    conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
    conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
    conv2 = (Conv2D(32, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv2)
        
    conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
    conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
    conv3 = (Conv2D(64, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv3)
        
    conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
    conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv4)
    conv4 = (Conv2D(128, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv4)
        
    re_shape = keras.layers.core.Reshape((-1, 4*128), input_shape=(-1, 4, 128))(conv4)
    _input_whisper = Input(shape=(None, 1024))
    bottleneck=TimeDistributed(Dense(512, activation='relu'))(_input_whisper)
    concat_with_whisper = concatenate([re_shape, bottleneck],axis=1) 
    blstm=Bidirectional(CuDNNLSTM(128, return_sequences=True), merge_mode='concat')(concat_with_whisper)

    flatten = TimeDistributed(keras.layers.core.Flatten())(blstm)
    dense1=TimeDistributed(Dense(128, activation='relu'))(flatten)
    dense1=Dropout(0.3)(dense1)
    
    attention = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,kernel_regularizer=keras.regularizers.l2(1e-4),bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4, name='Attention')(dense1)
    Frame_score=TimeDistributed(Dense(1, activation='sigmoid'), name='Frame_score')(attention)
    Qual_score=GlobalAveragePooling1D(name='Qual_score')(Frame_score)
    
    attention2 = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,kernel_regularizer=keras.regularizers.l2(1e-6),bias_regularizer=keras.regularizers.l1(1e-6),attention_regularizer_weight=1e-6, name='Attention2')(dense1)
    Frame_intell=TimeDistributed(Dense(1, activation='sigmoid'), name='Frame_intell')(attention2)
    Intell_score= GlobalAveragePooling1D(name='Intell_score')(Frame_intell)
    
    model = Model(outputs=[Qual_score, Frame_score, Intell_score , Frame_intell], inputs=[_input,_input_end2end, _input_whisper])
  
    return model

def train(train_list, train_list_whisper, NUM_TRAIN, valid_list, valid_list_whisper, NUM_VALID, pathmodel):    
    print ('model building...')
    model = BLSTM_CNN_with_ATT_cross_domain()
    adam = Adam(lr=1e-5)
    
    model.compile(loss={'Qual_score': 'mse', 'Frame_score': 'mse', 'Intell_score': 'mse', 'Frame_intell': 'mse'}, optimizer=adam)
    plot_model(model, to_file=pathmodel+'.png', show_shapes=True)
    
    with open(pathmodel+'.json','w') as f:    # save the model
        f.write(model.to_json()) 
    checkpointer = ModelCheckpoint(filepath=pathmodel+'.hdf5', verbose=1, save_best_only=True, mode='min')  
    
    print ('training...')
    g1 = data_generator(train_list, train_list_whisper)
    g2 = data_generator(valid_list, valid_list_whisper)

    hist=model.fit_generator(g1,steps_per_epoch=NUM_TRAIN, epochs=epoch, verbose=1,validation_data=g2,validation_steps=NUM_VALID,max_queue_size=1, workers=1,callbacks=[checkpointer])
               					
    model.save(pathmodel+'.h5')

    # plotting the learning curve
    TrainERR=hist.history['loss']
    ValidERR=hist.history['val_loss']
    print ('@%f, Minimun error:%f, at iteration: %i' % (hist.history['val_loss'][epoch-1], np.min(np.asarray(ValidERR)),np.argmin(np.asarray(ValidERR))+1))
    print ('drawing the training process...')
    plt.figure(2)
    plt.plot(range(1,epoch+1),TrainERR,'b',label='TrainERR')
    plt.plot(range(1,epoch+1),ValidERR,'r',label='ValidERR')
    plt.xlim([1,epoch])
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.grid(True)
    plt.show()
    plt.savefig('Learning_curve_'+pathmodel+'.png', dpi=150)


def test(Test_list_QI_score, Test_list_whisper,pathmodel):
    print ('testing...')
    model = BLSTM_CNN_with_ATT_cross_domain()
    model.load_weights(pathmodel+'.h5')

    Quality_Predict=np.zeros([len(Test_list_QI_score),])
    Quality_true   =np.zeros([len(Test_list_QI_score),])

    Intell_Predict=np.zeros([len(Test_list_QI_score),])
    Intell_true   =np.zeros([len(Test_list_QI_score),])
    
    path_whisper='/data1/user_ryandhimas/MTI_with_Whisper/Test_SSL_Feat_Med_Voice_MOS/'
    for i in range(len(Test_list_QI_score)):
       print i
       Asessment_filepath=Test_list_QI_score[i].split(',')

       noisy_LP, noisy_end2end =Sp_and_phase(Asessment_filepath[0])    

       noisy_whisper = np.load(Test_list_whisper[i])
       noisy_whisper =np.reshape(noisy_whisper,(1,noisy_whisper.shape[0],1024))
         
       quality=float(Asessment_filepath[1])         
       intell=float(Asessment_filepath[2])

       [Quality_1, frame_score, Intell_1, frame_intell]=model.predict([noisy_LP,noisy_end2end,noisy_whisper], verbose=0, batch_size=batch_size)

       denorm_Quality_predict = denorm(Quality_1)
       Quality_Predict[i]=denorm_Quality_predict
       Quality_true[i]   =quality

       Intell_Predict[i]=Intell_1
       Intell_true[i]   =intell
       
    MSE=np.mean((Quality_true-Quality_Predict)**2)
    print ('Test error= %f' % MSE)
    LCC=np.corrcoef(Quality_true, Quality_Predict)
    print ('Linear correlation coefficient= %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(Quality_true.T, Quality_Predict.T)
    print ('Spearman rank correlation coefficient= %f' % SRCC[0])
    
    # Plotting the scatter plot
    M=np.max([np.max(Quality_Predict),5])
    plt.figure(1)
    plt.scatter(Quality_true, Quality_Predict, s=14)
    plt.xlim([0,M])
    plt.ylim([0,M])
    plt.xlabel('True Quality')
    plt.ylabel('Predicted Quality')
    plt.title('LCC= %f, SRCC= %f, MSE= %f' % (LCC[0][1], SRCC[0], MSE))
    plt.show()
    plt.savefig('Scatter_plot_Qual_'+pathmodel+'_conf.png', dpi=150)

    MSE=np.mean((Intell_true-Intell_Predict)**2)
    print ('Test error= %f' % MSE)
    LCC=np.corrcoef(Intell_true, Intell_Predict)
    print ('Linear correlation coefficient= %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(Intell_true.T, Intell_Predict.T)
    print ('Spearman rank correlation coefficient= %f' % SRCC[0])

    # Plotting the scatter plot
    M=np.max([np.max(Intell_Predict),1])
    plt.figure(2)
    plt.scatter(Intell_true, Intell_Predict, s=14)
    plt.xlim([0,M])
    plt.ylim([0,M])
    plt.xlabel('True Intell')
    plt.ylabel('Predicted Intell')
    plt.title('LCC= %f, SRCC= %f, MSE= %f' % (LCC[0][1], SRCC[0], MSE))
    plt.show()
    plt.savefig('Scatter_plot_Intell_'+pathmodel+'_conf.png', dpi=150)
    
if __name__ == '__main__':  
     
    parser = argparse.ArgumentParser('')
    parser.add_argument('--gpus', type=str, default='0') 
    parser.add_argument('--mode', type=str, default='train') 
    
    args = parser.parse_args() 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    pathmodel="MOSA-Net_Plus_VoiceMOS"

    ########Format of Data######
    # The list of data looks like this ---> quality_score, intelligibility_score, path
    
    #################################################################             
    ######################### Training data #########################
    Train_list_wav = ListRead('TRAINSET.txt')
    Train_list_whisper = ListRead('TRAINSET_whisper.txt')
    NUM_DATA =  len(Train_list_wav)

    NUM_TRAIN = int(NUM_DATA*0.9) 
    NUM_VALID = NUM_DATA-NUM_TRAIN

    train_list= Train_list_wav[: NUM_TRAIN]
    random.Random(7).shuffle(train_list)
    valid_list= Train_list_wav[NUM_TRAIN: ]

    train_list_whisper= Train_list_whisper[: NUM_TRAIN]
    random.Random(7).shuffle(train_list_whisper)
    valid_list_whisper= Train_list_whisper[NUM_TRAIN: ]

    ################################################################
    ######################### Testing data #########################
    Test_list_QI_score=ListRead('TESTSET.txt')
    Test_list_whisper=ListRead('TESTSET_whisper.txt')       
    Num_testdata= len (Test_list_QI_score)

    if args.mode == 'train':
       print ('training')  
       train(train_list, train_list_whisper, NUM_TRAIN, valid_list, valid_list_whisper, NUM_VALID, pathmodel)
       print ('complete training stage')    
    
    if args.mode == 'test':      
       print ('testing') 
       test(Test_list_QI_score, Test_list_whisper,pathmodel)
       print ('complete testing stage')
