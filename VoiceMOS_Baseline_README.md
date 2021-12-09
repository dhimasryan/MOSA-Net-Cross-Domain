# MOSA-Net Cross-Domain: Baseline system of the first VoiceMOS Challenge 
Author: Ryandhimas Zezario (Academia Sinica) Email: ryandhimas@citi.sinica.edu.tw

The MOSA-Net Cross-Domain system implemented in this repository serves as one of the baselines of the first VoiceMOS Challenge, a challenge to compare different systems and approaches on the task of predicting the MOS score of synthetic speech. In this challenge, we use the BVCC dataset.

## Training phase (phase 1)
During the training phase, the training set and the developement set are released. In the following, we demonstrate how to train the model using the training set, and decode using the developement set to generate a result file that can be submitted to the CodaLab platform.

**Data preparation** 

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
