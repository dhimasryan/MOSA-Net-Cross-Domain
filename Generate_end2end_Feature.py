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
random.seed(999)


def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path
    
def Save_Npy_end2end(filepath,list_new):
    path = filepath [3]
    S=path.split('/')
    wave_name=S[-1]
    name = wave_name[:-4] 
    dir_ = S[-1]
    new_name =  name +'.npy'

    cached_path = os.path.join("/DIR/End2End_Waveform/"+dir_,new_name)
         
    signal, rate  = librosa.load(path,sr=16000)
    signal=signal/np.max(abs(signal))

    F = np.reshape(signal,(1,signal.shape[0],1))
    np.save(cached_path,F)
        
    info = filepath[0]+','+filepath[1]+','+filepath[2]+','+str(cached_path)
    list_new.append(info)
    
    return list_new

    
def extracting_end2end_features(file_list):
    list_new=[]
    for index in range(len(file_list)):
        filepath = file_list[index].split(',')
        list_new=Save_Npy_end2end(filepath, list_new)    
   
    with open('List_Npy_Train_end2end.txt','w') as g:
        for item in list_new:
          g.write("%s\n" % item)


if __name__ == '__main__':	
    print 'Extracting End2End Features...'	
    
    Enhanced_list = Get_filenames('EnhAllDataList_PESQ_STOI_SDI.txt')
    Noisy_list = Get_filenames('NoisyList_PESQ_STOI_SDI.txt')
    Clean_list = Get_filenames('CleanList_PESQ_STOI_SDI.txt')

    Enhanced_noisy_list=Enhanced_list+Noisy_list
    Train_list= Enhanced_noisy_list+Clean_list
    
    random.shuffle(Train_list)

    extracting_end2end_features(Train_list)
	
