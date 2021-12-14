# MOSA-Net Cross-Domain: Baseline system of the first VoiceMOS Challenge 
Author: Ryandhimas Zezario (Academia Sinica) Email: ryandhimas@citi.sinica.edu.tw

The MOSA-Net Cross-Domain system implemented in this repository serves as one of the baselines of the first VoiceMOS Challenge, a challenge to compare different systems and approaches on the task of predicting the MOS score of synthetic speech. In this challenge, we use the BVCC dataset.

## Training phase (phase 1)
During the training phase, the training set and the developement set are released. In the following, we demonstrate how to train the model using the training set, and decode using the developement set to generate a result file that can be submitted to the CodaLab platform.

 ### Data preparation ###

After downloading the dataset preparation scripts, please follow the instructions to gather the complete training and development set. For the rest of this README, we assume that the data is put under data/, but feel free to put it somewhere else. The data directorty should have the following structure:
```js
data
└── phase1-main
    ├── DATA
    │   ├── mydata_system.csv
    │   ├── sets
    │   │   ├── DEVSET
    │   │   ├── train_mos_list.txt
    │   │   ├── TRAINSET
    │   │   └── val_mos_list.txt
    │   └── wav
    └─── ...
 ```
 
 ### Training the Model  ###
 
Before training the model, please make sure that all dependencies have been installed correctly.

Please be noted, that the shared environment in ```./Environment/environment.yml``` is specifically used to run ```MOSA-Net_Cross_Domain_VoiceMOS_Challenge.py```. To generate Self Supervised Learning (SSL) feature ```Extracting_Hubert_Feature_VoiceMOS_Challenge.py```, please use ```python 3.6``` and follow the instructions in following <a href="https://github.com/pytorch/fairseq" target="_blank">link</a> to deploy fairseq module.
 
For extracting Self Supervised Learning (SSL) feature, please make sure that <a href="https://github.com/pytorch/fairseq" target="_blank">fairseq</a> module can be imported correctly and you can put fairseq under ```MOSA-Net-Cross-Domain/ ``` and put the Hubert model <a href="https://github.com/pytorch/fairseq/tree/main/examples/hubert#load-a-pretrained-model" target="_blank">(hubert_large_ll60k.pt)</a> under ```fairseq/```

You can use following script to extract Hubert-SSL feature.
```js
python Extracting_Hubert_Feature_VoiceMOS_Challenge.py --track phase1-main
 ```
 
 Next to train the MOSA-Net Cross-Domain model, please use the following scipt:
 ```js
python MOSA-Net_Cross_Domain_VoiceMOS_Challenge.py --gpus <assigned GPU> --name <Model Name> --track phase1-main --mode train
```

### Inference from pretrained model ###
For testing stage, you can use the following script:
```js
python MOSA-Net_Cross_Domain_VoiceMOS_Challenge.py --gpus <assigned GPU> --name <Model Name> --track phase1-main --mode test
```
Besides, by using the pretrained model ```./PreTrained_VoiceMOSChallenge/MOSA-Net_Cross_Domain_100epoch.h5```, you should get the following results.
```js
Utterance Level-Score
Test error= 0.277645
Linear correlation coefficient= 0.818193
Spearman rank correlation coefficient= 0.817553
Kendalls tau correlation= 0.632293

Systems Level-Score
Test error= 0.145215
Linear correlation coefficient= 0.903258
Spearman rank correlation coefficient= 0.899877
Kendalls tau correlation= 0.738427
complete testing stage
```
Additionally, the answer.txt file will also be generated


### Out-of-domain track (OOD) Track ###
For the OOD track, please make sure you have extracted the data correctly. We assume that the data is put under data/, but feel free to put it somewhere else.

For extracting Self Supervised Learning (SSL) feature, please use the following script:
```js
python Extracting_Hubert_Feature_VoiceMOS_Challenge.py --track phase1-ood
 ```
 
For training the MOSA-Net on OOD track, you can simply use the following script:
 ```js
python MOSA-Net_Cross_Domain_VoiceMOS_Challenge.py --gpus <assigned GPU> --name <Model Name> --track phase1-ood --mode train
```

Similar to the above command, we can get the inference score by using the following script:
```js
python MOSA-Net_Cross_Domain_VoiceMOS_Challenge.py --gpus <assigned GPU> --name <Model Name> --track phase1-ood --mode test
```

### Submission to CodaLab ###

The submission format of the CodaLab competition platform is a zip file (can be any name) containing a text file called answer.txt (this naming is a MUST).
To submit to the CodaLab competition platform, compress answer.txt in zip format (via zip command in Linux or GUI in MacOS) and name it whatever you want. Then this zip file is ready to be submitted! 

For the detailed instruction, please kindly check the following <a href="https://github.com/unilight/LDNet/blob/main/VoiceMOS_baseline_README.md#submission-generation-for-the-codalab-platform" target="_blank">link</a>.
