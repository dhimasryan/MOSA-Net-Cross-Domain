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
    
def Save_Npy_PS(filepath,list_new ):
    Noisy=False
    path = filepath [3]
    S=path.split('/')
    wave_name=S[-1]
    name = wave_name[:-4] 
    dir_ = S[-1]
    new_name =  name +'.npy'

    cached_path = os.path.join("/DIR/PS/"+dir_,new_name)
    
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
    np.save(cached_path,NLp)
        
    info = filepath[0]+','+filepath[1]+','+filepath[2]+','+str(cached_path)
    list_new.append(info)
    
    return list_new
    
def extracting_PS_features(file_list):
    list_new=[]
    for index in range(len(file_list)):
        filepath = file_list[index].split(',')
        list_new=Save_Npy_PS(filepath, list_new)    
   
    with open('List_Npy_Train_PS.txt','w') as g:
        for item in list_new:
          g.write("%s\n" % item)


if __name__ == '__main__':	
    print 'Extracting PS Features...'	
    
    Enhanced_list = Get_filenames('EnhAllDataList_PESQ_STOI_SDI.txt')
    Noisy_list = Get_filenames('NoisyList_PESQ_STOI_SDI.txt')
    Clean_list = Get_filenames('CleanList_PESQ_STOI_SDI.txt')

    Enhanced_noisy_list=Enhanced_list+Noisy_list
    Train_list= Enhanced_noisy_list+Clean_list
    
    random.shuffle(Train_list)

    extracting_PS_features(Train_list)
	
