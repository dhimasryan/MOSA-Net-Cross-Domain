"""
@author: Ryandhimas Zezario
ryandhimas@citi.sinica.edu.tw
"""

import os, sys
import keras
import matplotlib
import math
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Sequential, model_from_json, Model
from keras.layers import Layer, concatenate
from keras.layers.core import Dense, Dropout, Flatten, Activation, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.pooling import GlobalAveragePooling1D, AveragePooling1D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.backend import squeeze
from keras.layers import LSTM, TimeDistributed, Bidirectional, dot, Input, CuDNNLSTM
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

def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path

def Sp_and_phase(path, Noisy=False):

    audio_data, _ = librosa.load(path, sr=16000)
    audio_data=audio_data/np.max(abs(audio_data))   
    
    F = librosa.stft(audio_data,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
      
    Lp=np.abs(F)
    phase=np.angle(F)
    if Noisy==True:    
        meanR = np.mean(Lp, axis=1).reshape((257,1))
        stdR = np.std(Lp, axis=1).reshape((257,1))+1e-12
        NLp = (Lp-meanR)/stdR
    else:
        NLp=Lp
    
    NLp=np.reshape(NLp.T,(1,NLp.shape[1],257))
    end2end = np.reshape(audio_data,(1,audio_data.shape[0],1)) 
    return NLp, end2end

def train_data_generator(file_list, file_list_hubert, track_name):
	index=0
        data_path ='./data/'+track_name+'/DATA/wav/'
	while True:
         mos_filepath=file_list[index].split(',')
         hubert_filepath=file_list_hubert[index].split(',')
         
         complete_path = data_path + mos_filepath[0]
       
         noisy_LP,noisy_end2end =Sp_and_phase(complete_path)       
         noisy_hubert =np.load(hubert_filepath[1])    
                 
         mos=norm_data(np.asarray(float(mos_filepath[1])).reshape([1]))

         feat_length_end2end = math.ceil(float(noisy_end2end.shape[1])/256)
         final_len = noisy_LP.shape[1] + int(feat_length_end2end) + noisy_hubert.shape[1]
         
         index += 1
         if index == len(file_list):
             index = 0
            
             random.Random(7).shuffle(file_list)
             random.Random(7).shuffle(file_list_hubert)

         yield  [noisy_LP, noisy_end2end, noisy_hubert], [mos, mos[0]*np.ones([1,final_len,1])]
		 
def val_data_generator(file_list, file_list_hubert, track_name):
	index=0
        data_path ='./data/'+track_name+'/DATA/wav/'
	while True:
         mos_filepath=file_list[index].split(',')
         hubert_filepath=file_list_hubert[index].split(',')
         
         complete_path = data_path + mos_filepath[0]
         
         noisy_LP,noisy_end2end =Sp_and_phase(complete_path)       
         noisy_hubert =np.load(hubert_filepath[1])    
              
         mos=norm_data(np.asarray(float(mos_filepath[1])).reshape([1]))

         feat_length_end2end = math.ceil(float(noisy_end2end.shape[1])/256)
         final_len = noisy_LP.shape[1] + int(feat_length_end2end) + noisy_hubert.shape[1]
         
         index += 1
         if index == len(file_list):
             index = 0
            
             random.Random(7).shuffle(file_list)
             random.Random(7).shuffle(file_list_hubert)

         yield  [noisy_LP, noisy_end2end, noisy_hubert], [mos, mos[0]*np.ones([1,final_len,1])]

def norm_data(input_x):
    input_x = (input_x-0)/(5-0)
    return input_x
        
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
    _input_hubert = Input(shape=(None, 1024))
    mean_polling = AveragePooling1D(pool_size=2,strides=1, padding='same')(_input_hubert)
    bottleneck=TimeDistributed(Dense(512))(mean_polling)
    concat_with_wave2vec = concatenate([re_shape, bottleneck],axis=1) 
    blstm=Bidirectional(CuDNNLSTM(128, return_sequences=True), merge_mode='concat')(concat_with_wave2vec)

    flatten = TimeDistributed(keras.layers.core.Flatten())(blstm)
    dense1=TimeDistributed(Dense(128, activation='relu'))(flatten)
    dense1=Dropout(0.3)(dense1)

    attention = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,kernel_regularizer=keras.regularizers.l2(1e-4),bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4, name='Attention')(dense1)
    Frame_score=TimeDistributed(Dense(1, activation='sigmoid'), name='Frame_score')(attention)
    MOS_score=GlobalAveragePooling1D(name='MOS_score')(Frame_score)

    model = Model(outputs=[MOS_score, Frame_score], inputs=[_input,_input_end2end, _input_hubert])

    return model
    
def train(Train_list, Train_list_hubert, Num_train, Test_list, Test_list_hubert, Num_testdata, pathmodel, track_name):
    print 'model building...'
    
    model = BLSTM_CNN_with_ATT_cross_domain()
    
    get_model_name = pathmodel.split('/')  
    model_name =  get_model_name [2]
    
    adam = Adam(lr=1e-6)
    
    model.compile(loss={'MOS_score': 'mse', 'Frame_score': 'mse'}, optimizer=adam)
    plot_model(model, to_file='model_'+str(model_name)+'_epoch_'+str(epoch)+'.png', show_shapes=True)
 
    if track_name == 'phase1-main':
       print('Running Main-track')    
    else:
       print('Load Main-track model as initialized model for OOD-track')
       model.load_weights('./PreTrained_VoiceMOSChallenge/MOSA-Net_Cross_Domain_epoch_100.h5')    
       
    with open(pathmodel+'_epoch_'+str(epoch)+'.json','w') as f:    # save the model
        f.write(model.to_json()) 
    checkpointer = ModelCheckpoint(filepath=pathmodel+'_epoch_'+str(epoch)+'.hdf5', verbose=1, save_best_only=True, mode='min')  

    print 'training...'
    g1 = train_data_generator(Train_list, Train_list_hubert, track_name)
    g2 = val_data_generator  (Test_list, Test_list_hubert, track_name)

    hist=model.fit_generator(g1,steps_per_epoch=Num_train, epochs=epoch, verbose=1,validation_data=g2,validation_steps=Num_testdata,max_queue_size=1, workers=1,callbacks=[checkpointer])
               					
    model.save(pathmodel+'_epoch_'+str(epoch)+'.h5')

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
    plt.savefig('Learning_curve_'+str(model_name)+'_epoch_'+str(epoch)+'.png', dpi=150)


def Test(Test_list_PS, Test_list_hubert,pathmodel, track_name):
    print 'load model...'
    
    model_test = BLSTM_CNN_with_ATT_cross_domain()
    model_test.load_weights(pathmodel+'_epoch_'+str(epoch)+'.h5')

    get_model_name = pathmodel.split('/')  
    model_name =  get_model_name [2]
    
    print 'testing...'
    MOS_Predict=np.zeros([len(Test_List),])
    MOS_True   =np.zeros([len(Test_List),])

    data_path ='./data/'+track_name+'/DATA/wav/'
    list_predicted_mos_score =[]
    
    systems_list = list(set([item[0:8] for item in Test_List]))   
    MOS_Predict_systems = {system:[] for system in systems_list}
    MOS_True_systems = {system:[] for system in systems_list}
    
    for i in range(len(Test_List)): 
       print i
       Asessment_filepath=Test_List[i].split(',')
       hubert_filepath = Test_List_Hubert_feat[i].split(',')
       wav_name = Asessment_filepath[0]  
       
       complete_path = data_path + wav_name
       noisy_LP, noisy_end2end =Sp_and_phase(complete_path) 
       noisy_hubert = np.load(hubert_filepath[1])
 
       mos=float(Asessment_filepath[1])

       [MOS_1, frame_mos]=model_test.predict([noisy_LP,noisy_end2end,noisy_hubert], verbose=0, batch_size=batch_size)

       denorm_MOS_predict = denorm(MOS_1)
       MOS_Predict[i]=denorm_MOS_predict
       MOS_True[i]   =mos

       system_names = wav_name[0:8]
 
       MOS_Predict_systems[system_names].append(denorm_MOS_predict[0])
       MOS_True_systems[system_names].append(mos)
                
       estimated_score = denorm_MOS_predict[0]
       info = Asessment_filepath[0]+','+str(estimated_score[0])       
       list_predicted_mos_score.append(info)       

    MOS_Predict_systems = np.array([np.mean(scores) for scores in MOS_Predict_systems.values()])
    MOS_True_systems = np.array([np.mean(scores) for scores in MOS_True_systems.values()])
    
   
    with open('List_predicted_score_mos'+str(model_name)+'_epoch_'+str(epoch)+'_answer.txt','w') as g:
        for item in list_predicted_mos_score:
          g.write("%s\n" % item)

    print ('Utterance Level-Score')
    MSE=np.mean((MOS_True-MOS_Predict)**2)
    print ('Test error= %f' % MSE)
    LCC=np.corrcoef(MOS_True, MOS_Predict)
    print ('Linear correlation coefficient= %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(MOS_True.T, MOS_Predict.T)
    print ('Spearman rank correlation coefficient= %f' % SRCC[0])
    KTAU=scipy.stats.kendalltau(MOS_True, MOS_Predict)
    print ('Kendalls tau correlation= %f' % KTAU[0])    
    print ('')
    
    # Plotting the scatter plot
    M=np.max([np.max(MOS_Predict),5])
    plt.figure(1)
    plt.scatter(MOS_True, MOS_Predict, s=14)
    plt.xlim([0,M])
    plt.ylim([0,M])
    plt.xlabel('True MOS')
    plt.ylabel('Predicted MOS')
    plt.title('LCC= %f, SRCC= %f, MSE= %f, KTAU= %f' % (LCC[0][1], SRCC[0], MSE, KTAU[0]))
    plt.show()
    plt.savefig('Scatter_plot_MOS_'+str(model_name)+'_epoch_'+str(epoch)+'.png', dpi=150)

    print ('Systems Level-Score')
    MSE=np.mean((MOS_True_systems-MOS_Predict_systems)**2)
    print ('Test error= %f' % MSE)
    LCC=np.corrcoef(MOS_True_systems, MOS_Predict_systems)
    print ('Linear correlation coefficient= %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(MOS_True_systems.T, MOS_Predict_systems.T)
    print ('Spearman rank correlation coefficient= %f' % SRCC[0])
    KTAU=scipy.stats.kendalltau(MOS_True_systems, MOS_Predict_systems)
    print ('Kendalls tau correlation= %f' % KTAU[0])    

if __name__ == '__main__':  
    parser = argparse.ArgumentParser('')
    parser.add_argument('--gpus', type=str, default='0') 
    parser.add_argument('--name', type=str, default='MOSA-Net_Cross_Domain-main')     
    parser.add_argument('--track', type=str, default='phase1-main')     
    parser.add_argument('--mode', type=str, default='train') 
    args = parser.parse_args() 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    pathmodel='./PreTrained_VoiceMOSChallenge/'+str(args.name)
    track_name = args.track
    
    Train_list = ListRead('./data/'+track_name+'/DATA/sets/train_mos_list.txt')
    Train_list_hubert = ListRead('./data/'+track_name+'/DATA/sets/List_Npy_Train_hubert_MOS_Challenge_phase1_main.txt')
    NUM_DATA_TRAIN =  len(Train_list)
    
    random.Random(7).shuffle(Train_list)
    random.Random(7).shuffle(Train_list_hubert)

    Val_list = ListRead('./data/'+track_name+'/DATA/sets/val_mos_list.txt')
    Val_list_hubert = ListRead('./data/'+track_name+'/DATA/sets/List_Npy_Val_hubert_MOS_Challenge_phase1_main.txt')
    NUM_DATA_VAL =  len(Val_list) 
    
    Test_List=Val_list
    Test_List_Hubert_feat=Val_list_hubert

    if args.mode == 'train':
       print 'training' 
       train(Train_list, Train_list_hubert, NUM_DATA_TRAIN, Val_list, Val_list_hubert, NUM_DATA_VAL, pathmodel, track_name)
       print 'complete training stage'    
    if args.mode == 'test':    
       print 'testing' 
       Test(Test_List, Test_List_Hubert_feat,pathmodel, track_name)
       print 'complete testing stage'