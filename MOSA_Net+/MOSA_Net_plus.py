# ===================================================================================================
# MOSA-Net pytorch was written by Ryandhimas Zezario
# ====================================================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import argparse
from transformers import AutoFeatureExtractor, WhisperModel
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import librosa
import random
import pdb
import numpy as np
import math
import speechbrain
from tqdm import tqdm
import multiprocessing
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
random.seed(999)

random.seed(999)

   
class MosPredictor(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.mean_net_conv = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(128),
            nn.ReLU())
        
        self.relu_ = nn.ReLU()
        self.sigmoid_ = nn.Sigmoid()
        
        self.ssl_features = 1280
        self.dim_layer = nn.Linear(self.ssl_features, 512)

        self.mean_net_rnn = nn.LSTM(input_size = 512, hidden_size = 128, num_layers = 1, batch_first = True, bidirectional = True)
        self.mean_net_dnn = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )        

        self.sinc = speechbrain.nnet.CNN.SincConv(in_channels=1, out_channels=257, kernel_size=251, stride=256, sample_rate=16000)
        self.att_output_layer_quality = nn.MultiheadAttention(128, num_heads=8)                
        self.output_layer_quality = nn.Linear(128, 1)
        self.qualaverage_score = nn.AdaptiveAvgPool1d(1)  
     
        self.att_output_layer_intell = nn.MultiheadAttention(128, num_heads=8)           
        self.output_layer_intell = nn.Linear(128, 1)
        self.intellaverage_score = nn.AdaptiveAvgPool1d(1)  
                       
        self.att_output_layer_stoi= nn.MultiheadAttention(128, num_heads=8)          
        self.output_layer_stoi = nn.Linear(128, 1)        
        self.stoiaverage_score = nn.AdaptiveAvgPool1d(1) 

    def new_method(self):
        self.sin_conv 
                
    def forward(self, wav, lps, whisper):
        #SSL Features
        wav_ = wav.squeeze(1)  ## [batches, audio_len]
        ssl_feat_red = self.dim_layer(whisper.squeeze(1))
        ssl_feat_red = self.relu_(ssl_feat_red)
 
        #PS Features
        sinc_feat=self.sinc(wav.squeeze(1))
        unsq_sinc =  torch.unsqueeze(sinc_feat, axis=1)
        concat_lps_sinc = torch.cat((lps,unsq_sinc), axis=2)
        cnn_out = self.mean_net_conv(concat_lps_sinc)
        batch = concat_lps_sinc.shape[0]
        time = concat_lps_sinc.shape[2]        
        re_cnn = cnn_out.view((batch, time, 512))
        
        concat_feat = torch.cat((re_cnn,ssl_feat_red), axis=1)
        out_lstm, (h, c) = self.mean_net_rnn(concat_feat)
        out_dense = self.mean_net_dnn(out_lstm) # (batch, seq, 1)       
        
        quality_att, _ = self.att_output_layer_quality (out_dense, out_dense, out_dense) 
        frame_quality = self.output_layer_quality(quality_att)
        frame_quality = self.sigmoid_(frame_quality)   
        quality_utt = self.qualaverage_score(frame_quality.permute(0,2,1))

        int_att, _ = self.att_output_layer_intell (out_dense, out_dense, out_dense) 
        frame_int = self.output_layer_intell(int_att)
        frame_int = self.sigmoid_(frame_int)   
        int_utt = self.intellaverage_score(frame_int.permute(0,2,1))

                
        return quality_utt.squeeze(1), int_utt.squeeze(1), frame_quality.squeeze(2), frame_int.squeeze(2)


class MyDataset(Dataset):
    def __init__(self, mos_list, whisper_list) :
        self.wav_file = open(mos_list, 'r').read().splitlines()
        self.whisper_list = open(whisper_list, 'r').read().splitlines()
        
        
    def __getitem__(self, idx):             
        wav_file = self.wav_file[idx]
        wav_file = wav_file.split(',')
        wavpath = wav_file[2]
        wav = torchaudio.load(wavpath)[0]     
        lps = torch.from_numpy(np.expand_dims(np.abs(librosa.stft(wav[0].detach().numpy(), n_fft = 512, hop_length=256,win_length=512)).T, axis=0))
        whisper = np.load(self.whisper_list[idx])
                          
        score_quality = torch.tensor(float(wav_file[0]))
        score_intell = torch.tensor(float(wav_file[1]))
                    
        return wav, lps, whisper, score_quality, score_intell

    def __len__(self):
        return len(self.wav_file)

def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path
    
def denorm(input_x):
    input_x = input_x*(5-0) + 0
    return input_x
    
def frame_score(y_true, y_predict):
    B,T = y_predict.size()  
    y_true_repeat = y_true.unsqueeze(1).repeat(1,T) #(B,T)  
    return y_true_repeat

def train(finetune_from_checkpoint,outdir):
    ckptdir = outdir
    my_checkpoint = finetune_from_checkpoint
    
    if not os.path.exists(ckptdir):
        os.system('mkdir -p ' + ckptdir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    trainlist = '/MOSA-Net_Plus_Torch/Train_VoiceMOS_2023.txt'
    validlist = '/MOSA-Net_Plus_Torch/Val_VoiceMOS_2023.txt'
    
    whisper_trainlist = '/MOSA-Net_Plus_Torch/Whisperfeat_Train_VoiceMOS_2023.txt'
    whisper_vallist = '/MOSA-Net_Plus_Torch/Whisperfeat_Val_VoiceMOS_2023.txt'

    
    trainset = MyDataset(trainlist, whisper_trainlist)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2) #collate_fn=trainset.collate_fn) # , collate_fn=lambda x: custom_collate(x, model_whisper, feat_ext_whisper))

    validset = MyDataset(validlist, whisper_vallist)
    validloader = DataLoader(validset, batch_size=1, shuffle=True, num_workers=2) #collate_fn=validset.collate_fn) # , collate_fn=lambda x: custom_collate(x, model_whisper, feat_ext_whisper))

    net = MosPredictor()
    net = net.to(device)

    if my_checkpoint != None:  ## do (further) finetuning
        net.load_state_dict(torch.load(my_checkpoint))
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-5)

    PREV_VAL_LOSS=9999999999
    orig_patience=5
    patience=orig_patience
    for epoch in range(1,26):
        STEPS=0
        net.train()
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader), 0):

            inputs, lps, whisper, labels_quality, labels_intell = data
            inputs = inputs.to(device)
            lps = lps.to(device)      
            whisper = whisper.to(device)      
            labels_quality = labels_quality.to(device)
            labels_intell = labels_intell.to(device)          
            
            optimizer.zero_grad()            
            output_quality, output_intell, frame_quality, frame_intell = net(inputs,lps,whisper)      
            
            label_frame_quality = frame_score(labels_quality, frame_quality)
            label_frame_intell = frame_score(labels_intell, frame_intell)

            loss_frame_quality = criterion(frame_quality, label_frame_quality)
            loss_frame_intell = criterion(frame_intell, label_frame_intell)
                                                  
            loss_quality = criterion(output_quality.squeeze(1), labels_quality)
            loss_intell = criterion(output_intell.squeeze(1), labels_intell)
                                  
            loss = loss_quality + loss_frame_quality + loss_intell + loss_frame_intell

            loss.backward()
            optimizer.step()
            STEPS += 1
            running_loss += loss.item()
        
        print('EPOCH: ' + str(epoch))
        print('AVG EPOCH TRAIN LOSS: ' + str(running_loss / STEPS))
        epoch_val_loss = 0.0
        net.eval()
        ## clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.memory_allocated()
            torch.cuda.synchronize() 
        
        ## validation
        VALSTEPS=0
        for i, data in enumerate(validloader, 0):            
            inputs, lps, whisper, labels_quality, labels_intell = data
            inputs = inputs.to(device)
            lps = lps.to(device)  
            whisper = whisper.to(device)
            labels_quality = labels_quality.to(device)
            labels_intell = labels_intell.to(device)
            
            output_quality, output_intell,  frame_quality, frame_intell = net(inputs,lps, whisper)
            
            label_frame_quality = frame_score(labels_quality, frame_quality)
            label_frame_intell = frame_score(labels_intell, frame_intell)

            loss_frame_quality = criterion(frame_quality, label_frame_quality)
            loss_frame_intell = criterion(frame_intell, label_frame_intell)
                                                  
            loss_quality = criterion(output_quality.squeeze(1), labels_quality)
            loss_intell = criterion(output_intell.squeeze(1), labels_intell)                    
            loss = loss_quality + loss_frame_quality + loss_intell + loss_frame_intell
                        
            epoch_val_loss += loss.item()
            VALSTEPS+=1


        avg_val_loss=epoch_val_loss/VALSTEPS    
        print('EPOCH VAL LOSS: ' + str(avg_val_loss))
        if avg_val_loss < PREV_VAL_LOSS:
            print('Loss has decreased')
            PREV_VAL_LOSS=avg_val_loss
            PATH = os.path.join(ckptdir, 'ckpt_' + str(epoch))
            torch.save(net.state_dict(), PATH)
            patience = orig_patience
        else:
            patience-=1
            if patience == 0:
                print('loss has not decreased for ' + str(orig_patience) + ' epochs; early stopping at epoch ' + str(epoch))
                break
        
    print('Finished Training')

def testing(my_checkpoint):
    checkpoint_split =  my_checkpoint.split('/')
    checkpoint_split = checkpoint_split [4]
    model_name = checkpoint_split [12:]
    
    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MosPredictor().to(device)
    model.eval()

    model.load_state_dict(torch.load(my_checkpoint))

    Test_List=ListRead('/MOSA-Net_Plus_Torch/Test_VoiceMOS_2023.txt')
    Test_Whisper_dir = '/MOSA-Net_Plus_Torch/Test_SSL_Feat_Whisperv3'

    Quality_Predict=np.zeros([len(Test_List),])
    Quality_true   =np.zeros([len(Test_List),])

    Intell_Predict=np.zeros([len(Test_List),])
    Intell_true   =np.zeros([len(Test_List),])
    
    print('Starting prediction')
    i = 0
    for path in tqdm(Test_List): 
              
        Asessment_filepath=path.split(',')
        
        wavefile = Asessment_filepath[0] 
        wavename_with_path = wavefile.split('/')
        wavename =  wavename_with_path [-1]  
        wavename = wavename [:-3] + 'npy'  
        
        quality = float(Asessment_filepath[1])
        intell = float(Asessment_filepath[2]) 
        
        wav = torchaudio.load(wavefile)[0] 
        lps = torch.from_numpy(np.expand_dims(np.abs(librosa.stft(wav[0].detach().numpy(), n_fft = 512, hop_length=256,win_length=512)).T, axis=0))
        lps = lps.unsqueeze(1)
        
        whisper = np.load(os.path.join(Test_Whisper_dir, wavename))
        wav = wav.to(device)
        lps = lps.to(device)
        whisper = torch.tensor(whisper)
        whisper = whisper.to(device)

        Quality_1, Intell_1, frame1, frame2 = model(wav,lps, whisper)

        quality_pred = Quality_1.cpu().detach().numpy()[0] 
        intell_pred = Intell_1.cpu().detach().numpy()[0]         
        
        Quality_Predict[i]=denorm(quality_pred)
        Quality_true[i]   =quality

        Intell_Predict[i]=intell_pred
        Intell_true[i]   =intell
        
        i+=1

    MSE=np.mean((Quality_true-Quality_Predict)**2)
    print ('Test error= %f' % MSE)
    LCC=np.corrcoef(Quality_true, Quality_Predict)
    print ('Linear correlation coefficient= %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(Quality_true.T, Quality_Predict.T)
    print ('Spearman rank correlation coefficient= %f' % SRCC[0])

    # Plotting the scatter plot
    M=np.max([np.max(Quality_Predict),1])
    plt.figure(1)
    plt.scatter(Quality_true, Quality_Predict, s=14)
    plt.xlim([0,M])
    plt.ylim([0,M])
    plt.xlabel('True Quality')
    plt.ylabel('Predicted Quality')
    plt.title('LCC= %f, SRCC= %f, MSE= %f' % (LCC[0][1], SRCC[0], MSE))
    plt.show()
    plt.savefig('Scatter_plot_Quality_'+model_name+'.png', dpi=150) 


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
    plt.savefig('Scatter_plot_Intell_'+model_name+'.png', dpi=150)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--gpus', type=str, default='5') 
    parser.add_argument('--mode', type=str, default='train') 
    parser.add_argument('--finetune_from_checkpoint', type=str, required=False, help='Path to your checkpoint to finetune from')
    parser.add_argument('--outdir', type=str, required=False, default='/MOSA-Net_Plus_Torch/Checkpoint', help='Output directory for your trained checkpoints')
    args = parser.parse_args()
        
    args = parser.parse_args() 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    if args.mode == 'train':
        train(args.finetune_from_checkpoint,args.outdir)
    else:
        model = args.outdir+'/ckpt_mosa_net_plus'
        testing(model)   
