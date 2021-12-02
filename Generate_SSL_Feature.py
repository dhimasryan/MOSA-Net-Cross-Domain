"""
@author: Ryandhimas Zezario
ryandhimas@citi.sinica.edu.tw
"""

import sys
import scipy.io
import librosa
import os
import time  
import numpy as np
import numpy.matlib
import random
import pdb
import torch
import fairseq


def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path
        
def Save_Npy_SSL(filepath, model, list_new ):
    path = filepath [3]
    S=path.split('/')
    wave_name=S[-1]
    name = wave_name[:-4] 
    dir_ = S[-2]
    new_name =  name +'.npy'

    cached_path = os.path.join("/DIR/SSL/"+dir_,new_name)
         
    signal = np.load(path)
    F = np.reshape(signal,(1,signal.shape[1]))
    F = torch.from_numpy(F).to("cuda:7")
    features = model(F, features_only=True, mask=False)['x']  
    causal = features.detach().to("cpu").numpy()

    np.save(cached_path,causal)
        
    info = filepath[0]+','+filepath[1]+','+filepath[2]+','+str(cached_path)
    list_new.append(info)
    
    return list_new
      
def train_data_generator(file_list,model):
    list_new=[]
    for index in range(len(file_list)):
        filepath = file_list[index].split(',')
        list_new=Save_Npy_SSL(filepath, model, list_new)    
   
    with open('List_Npy_Train_hubert_large_model.txt','w') as g:
        for item in list_new:
          g.write("%s\n" % item)

def extracting_SSL_features(Train_data):
    cp_path = '/data1/user_ryandhimas/fairseq/hubert_large_ll60k.pt'
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path], arg_overrides={"data": "/data1/user_ryandhimas/fairseq/fairseq/data/dict"})
    model = model[0]
    model.eval()
    device = torch.device("cuda:7")
    model = model.to(device)
    train_data_generator(Train_data,model)

if __name__ == '__main__':	
    print 'Extracting SSL Features...'	
    
    Enhanced_list = Get_filenames('EnhAllDataList_PESQ_STOI_SDI.txt')
    Noisy_list = Get_filenames('NoisyList_PESQ_STOI_SDI.txt')
    Clean_list = Get_filenames('CleanList_PESQ_STOI_SDI.txt')

    Enhanced_noisy_list=Enhanced_list+Noisy_list
    Train_list= Enhanced_noisy_list+Clean_list
    
    random.shuffle(Train_list)

    extracting_SSL_features(Train_list)
