"""
@author: Ryandhimas Zezario
ryandhimas@citi.sinica.edu.tw
"""

import os, sys
import keras
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Sequential, model_from_json, Model
from keras.layers import Layer
from keras.layers.core import Dense, Dropout, Flatten, Activation, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.backend import squeeze
from keras.layers import LSTM, TimeDistributed, Bidirectional, dot, Input
from keras.constraints import max_norm
from keras_self_attention import SeqSelfAttention
from SincNet import Sinc_Conv_Layer
import tensorflow as tf
import scipy.io
import scipy.stats
import librosa
import time  
import numpy as np
import numpy.matlib
import random
import argparse
import pdb
random.seed(999)

epoch=100
batch_size=1

def Get_filenames(ListPath):
    FileList=[];
    with open(ListPath) as fp:
        for line in fp:
            FileList.append(line.strip("\n"));
    return FileList;
    
def Feature_Extrator(path, Noisy=False):
    
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
    return NLp, phase

def train_data_generator(file_list_PS, file_list_End2End, file_list_hubert):
	index=0
	while True:
         PS_filepath=file_list_PS[index].split(',')
         end2end_filepath=file_list_End2End[index].split(',')
         hubert_filepath=file_list_hubert[index].split(',')
         noisy_LP =np.load(PS_filepath[3]) 
         noisy_end2end =np.load(end2end_filepath[3]) 
         noisy_hubert =np.load(hubert_filepath[3])          
         pesq=np.asarray(float(PS_filepath[0])).reshape([1])
         stoi=np.asarray(float(PS_filepath[1])).reshape([1])
         sdi=np.asarray(float(PS_filepath[2])).reshape([1])

         feat_length_end2end = math.ceil(float(noisy_end2end.shape[1])/256)
         final_len = noisy_LP.shape[1] + int(feat_length_end2end) + noisy_hubert.shape[1]
         
         index += 1
         if index == len(file_list_PS):
             index = 0
            
             random.Random(7).shuffle(file_list_PS)
             random.Random(7).shuffle(file_list_End2End)
             random.Random(7).shuffle(file_list_hubert)

         yield  [noisy_LP, noisy_end2end, noisy_hubert], [pesq, pesq[0]*np.ones([1,final_len,1]), stoi, stoi[0]*np.ones([1,final_len,1]), sdi, sdi[0]*np.ones([1,final_len,1])]

def val_data_generator(file_list_PS, file_list_End2End, file_list_hubert):
	index=0
	while True:
         PS_filepath=file_list_PS[index].split(',')
         end2end_filepath=file_list_End2End[index].split(',')
         hubert_filepath=file_list_hubert[index].split(',')
         noisy_LP =np.load(PS_filepath[3]) 
         noisy_end2end =np.load(end2end_filepath[3]) 
         noisy_hubert =np.load(hubert_filepath[3])          
         pesq=np.asarray(float(PS_filepath[0])).reshape([1])
         stoi=np.asarray(float(PS_filepath[1])).reshape([1])
         sdi=np.asarray(float(PS_filepath[2])).reshape([1])

         feat_length_end2end = math.ceil(float(noisy_end2end.shape[1])/256)
         final_len = noisy_LP.shape[1] + int(feat_length_end2end) + noisy_hubert.shape[1]
         
         index += 1
         if index == len(file_list_PS):
             index = 0
            
             random.Random(7).shuffle(file_list_PS)
             random.Random(7).shuffle(file_list_End2End)
             random.Random(7).shuffle(file_list_hubert)

         yield  [noisy_LP, noisy_end2end, noisy_hubert], [pesq, pesq[0]*np.ones([1,final_len,1]), stoi, stoi[0]*np.ones([1,final_len,1]), sdi, sdi[0]*np.ones([1,final_len,1])]

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
    
def train(Train_list_PS, Train_list_End2End, Train_list_hubert, Num_train, Test_list_PS, Test_list_End2End, Test_list_hubert, Num_testdata, pathmodel):
    print 'model building...'
    
    model = BLSTM_CNN_with_ATT_cross_domain()
    
    adam = Adam(lr=1e-4)
    model.compile(loss={'PESQ_score': 'mse', 'Frame_score': 'mse', 'STOI_score': 'mse', 'Frame_stoi': 'mse', 'SDI_score': 'mse','Frame_sdi': 'mse'}, optimizer=adam)
    plot_model(model, to_file='model_'+pathmodel+'.png', show_shapes=True)
    
    with open(pathmodel+'.json','w') as f:    # save the model
        f.write(model.to_json()) 
    checkpointer = ModelCheckpoint(filepath=pathmodel+'.hdf5', verbose=1, save_best_only=True, mode='min')  

    print 'training...'
    g1 = train_data_generator(Train_list_PS, Train_list_End2End, Train_list_hubert)
    g2 = val_data_generator  (Test_list_PS, Test_list_End2End, Test_list_hubert)

    hist=model.fit_generator(g1,steps_per_epoch=Num_train, epochs=epoch, verbose=1,validation_data=g2,validation_steps=Num_testdata,max_queue_size=1, workers=1,callbacks=[checkpointer])
               					
    model.save(pathmodel+'.h5')

    # plotting the learning curve
    TrainERR=hist.history['loss']
    ValidERR=hist.history['val_loss']
    print ('@%f, Minimun error:%f, at iteration: %i' % (hist.history['val_loss'][epoch-1], np.min(np.asarray(ValidERR)),np.argmin(np.asarray(ValidERR))+1))
    print 'drawing the training process...'
    plt.figure(2)
    plt.plot(range(1,epoch+1),TrainERR,'b',label='TrainERR')
    plt.plot(range(1,epoch+1),ValidERR,'r',label='ValidERR')
    plt.xlim([1,epoch])
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.grid(True)
    plt.show()
    plt.savefig('Learning_curve_MOSA-Net_Cross_Domain.png', dpi=150)


def Test(Test_list_PS, Test_list_End2End, Test_list_hubert,pathmodel):
    print 'load model...'
    
    model = BLSTM_CNN_with_ATT_cross_domain()
    model.load_weights(pathmodel+'.h5')
    
    print 'testing...'
    PESQ_Predict=np.zeros([len(Test_list_PS),])
    PESQ_true   =np.zeros([len(Test_list_PS),])
    STOI_Predict=np.zeros([len(Test_list_PS),])
    STOI_true   =np.zeros([len(Test_list_PS),])
    SDI_Predict=np.zeros([len(Test_list_PS),])
    SDI_true   =np.zeros([len(Test_list_PS),])
    
    for i in range(len(Test_list_PS)):
        PS_filepath=Test_list_PS[i].split(',')
        end2end_filepath=Test_list_End2End[i].split(',')
        hubert_filepath=Test_list_hubert[i].split(',')
        noisy_LP =np.load(PS_filepath[3])  
        noisy_end2end =np.load(end2end_filepath[3]) 
        noisy_hubert =np.load(hubert_filepath[3])      
        pesq=float(PS_filepath[0])
        stoi=float(PS_filepath[1])
        sdi=float(PS_filepath[2])

        [PESQ_score, frame_score, STOI_score, frame_stoi, SDI_score, frame_sdi]=model.predict([noisy_LP,noisy_end2end,noisy_hubert], verbose=0, batch_size=batch_size)
        #pdb.set_trace()
        PESQ_Predict[i]=PESQ_score
        PESQ_true[i]   =pesq

        STOI_Predict[i]=STOI_score
        STOI_true[i]   =stoi
	
        SDI_Predict[i]=SDI_score
        SDI_true[i]   =sdi
    

    MSE=np.mean((PESQ_true-PESQ_Predict)**2)
    print ('Test error= %f' % MSE)
    LCC=np.corrcoef(PESQ_true, PESQ_Predict)
    print ('Linear correlation coefficient= %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(PESQ_true.T, PESQ_Predict.T)
    print ('Spearman rank correlation coefficient= %f' % SRCC[0])

    # Plotting the scatter plot PESQ
    M=np.max([np.max(PESQ_Predict),4.5])
    plt.figure(1)
    plt.scatter(PESQ_true, PESQ_Predict, s=14)
    plt.xlim([0,M])
    plt.ylim([0,M])
    plt.xlabel('True PESQ')
    plt.ylabel('Predicted PESQ')
    plt.title('LCC= %f, SRCC= %f, MSE= %f' % (LCC[0][1], SRCC[0], MSE))
    plt.show()
    plt.savefig('Scatter_plot_PESQ_MOSA_Net_Cross_Domain.png', dpi=150)

    MSE=np.mean((STOI_true-STOI_Predict)**2)
    print ('Test error= %f' % MSE)
    LCC=np.corrcoef(STOI_true, STOI_Predict)
    print ('Linear correlation coefficient= %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(STOI_true.T, STOI_Predict.T)
    print ('Spearman rank correlation coefficient= %f' % SRCC[0])

    # Plotting the scatter plot STOI
    M=np.max([np.max(STOI_Predict),1])
    plt.figure(2)
    plt.scatter(STOI_true, STOI_Predict, s=14)
    plt.xlim([0,M])
    plt.ylim([0,M])
    plt.xlabel('True STOI')
    plt.ylabel('Predicted STOI')
    plt.title('LCC= %f, SRCC= %f, MSE= %f' % (LCC[0][1], SRCC[0], MSE))
    plt.show()
    plt.savefig('Scatter_plot_STOI_MOSA_Net_Cross_Domain.png', dpi=150)

    MSE=np.mean((SDI_true-SDI_Predict)**2)
    print ('Test error= %f' % MSE)
    LCC=np.corrcoef(SDI_true, SDI_Predict)
    print ('Linear correlation coefficient= %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(SDI_true.T, SDI_Predict.T)
    print ('Spearman rank correlation coefficient= %f' % SRCC[0])

    # Plotting the scatter plot
    M=np.max([np.max(SDI_Predict),1.5])
    plt.figure(3)
    plt.scatter(SDI_true, SDI_Predict, s=14)
    plt.xlim([0,M])
    plt.ylim([0,M])
    plt.xlabel('True SDI')
    plt.ylabel('Predicted SDI')
    plt.title('LCC= %f, SRCC= %f, MSE= %f' % (LCC[0][1], SRCC[0], MSE))
    plt.show()
    plt.savefig('Scatter_plot_SDI_MOSA_Net_Cross_Domain.png', dpi=150)


if __name__ == '__main__':  
     
    parser = argparse.ArgumentParser('')
    parser.add_argument('--gpus', type=str, default='0') 
    args = parser.parse_args() 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    pathmodel="MOSA-Net_Cross_Domain"

    Train_list_PS = ListRead('List_Npy_Train_PS.txt')
    Train_list_End2End = ListRead('List_Npy_Train_end2end.txt')
    Train_list_hubert = ListRead('List_Npy_Train_hubert_large_model.txt')
    Num_train =  len(Train_list_PS)

    Test_list_PS = ListRead('List_Npy_Test_PS.txt')
    Test_list_End2End = ListRead('List_Npy_Test_end2end.txt')
    Test_list_hubert = ListRead('List_Npy_Test_hubert_large_model.txt')
    Num_testdata= len (Test_list_PS)
 
    train(Train_list_PS, Train_list_End2End, Train_list_hubert, Num_train, Test_list_PS, Test_list_End2End, Test_list_hubert, Num_testdata, pathmodel)
    
    print 'testing' 
    Test(Test_list_PS, Test_list_End2End, Test_list_hubert,pathmodel)
    print 'complete testing stage'