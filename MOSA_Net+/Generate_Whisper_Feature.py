"""
@author: Ryandhimas Zezario
ryandhimas@citi.sinica.edu.tw
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import whisper
import torch
import argparse
import numpy as np 

def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path
    
def extract(input_list, filename, directory, mode):
    model = whisper.load_model("medium")
    device = torch.device("cuda")
    model.to(device)    

    list_new=[]      
    for i in range(len(input_list)):   
        Asessment_filepath=input_list[i]
        
        wavefile = Asessment_filepath               
        path_name = wavefile
        S=path_name.split('/')
        wave_name=S[-1]
        name = wave_name[:-4] 
        new_name =  name +'.npy' 
               
        cached_path = os.path.join(directory,new_name)
   
        audio = whisper.load_audio(wavefile)
        audio = whisper.pad_or_trim(audio)
    
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)
 
        feat=  result.audio_features
        feat = feat.detach().to("cpu").numpy()
        np.save(cached_path,feat)        
        info = str(cached_path)
        list_new.append(info)
    
    with open(filename+'.txt','w') as g:
        for item in list_new:
          g.write("%s\n" % item) 
          
          
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')     
    parser.add_argument('--filename', type=str, default='List_SSL_Whisper_Train_VoiceMOS')       
    args = parser.parse_args()
      
    print('Starting prediction')
    list_new =[]   

    if args.mode =="train":
       directory ='/Train_SSL_Feat_Med/'
       if not os.path.exists(directory):
          os.system('mkdir -p ' + directory)  
       
       Input_List=ListRead('Train_data.txt')   
       extract(Input_List, args.filename, directory, args.mode)  
       
    else :
       directory ='/Test_SSL_Feat_Med/'    
       if not os.path.exists(directory):
          os.system('mkdir -p ' + directory)  
       
       Input_List=ListRead('/data1/user_ryandhimas/MTI_with_Whisper/Lists/track3_run.txt')    
       extract(Input_List, args.filename, directory, args.mode)       
