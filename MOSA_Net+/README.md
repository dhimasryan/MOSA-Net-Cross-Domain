# MOSA-Net+ : An improved version of MOSA-Net 

### Abstract
This research introduces an enhanced version of the multi-objective speech assessment model--MOSA-Net+, by leveraging the acoustic features from Whisper, a large-scaled weakly supervised model. We first investigate the effectiveness of Whisper in deploying a more robust speech assessment model. After that, we explore combining representations from Whisper and SSL models.  The experimental results reveal that Whisper's embedding features can contribute to more accurate prediction performance. Moreover, combining the embedding features from Whisper and SSL models only leads to marginal improvement. As compared to intrusive methods, MOSA-Net, and other SSL-based speech assessment models, MOSA-Net+ yields notable improvements in estimating subjective quality and intelligibility scores across all evaluation metrics in Taiwan Mandarin Hearing In Noise test - Quality & Intelligibility (TMHINT-QI) dataset. To further validate its robustness, MOSA-Net+ was tested in the noisy-and-enhanced track of the VoiceMOS Challenge 2023, where it obtained the top-ranked performance among nine systems.

### Installation ###

You can download our environmental setup at ./Environment folder and use the following script for the installation.
```js
conda env create -f env.yml
```

### How to run the code ###

To extract the Whisper features, please use the following command.
```js
python Generate_Whisper_Feature.py --mode test --filename Whisperfeat_Test_VoiceMOS_2023
```
Please use following script to train the model:
```js
python MOSA_Net_plus.py --gpus <assigned GPU> --mode train
```
For, the testing stage, plase use the following script:
```js
python MOSA_Net_plus.py --gpus <assigned GPU> --mode test
```

### Citation ###

Please kindly cite our paper, if you find this code is useful.

<a id="1"></a> 
R. E. Zezario, S. -W. Fu, F. Chen, C. -S. Fuh, H. -M. Wang and Y. Tsao, "Deep Learning-Based Non-Intrusive Multi-Objective Speech Assessment Model With Cross-Domain Features," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 54-70, 2023, doi: 10.1109/TASLP.2022.3205757.

R. E. Zezario, Y.-W. Chen, S.-W. Fu, Y. Tsao, H.-M. Wang, C.-S. Fuh, "A Study on Incorporating Whisper for Robust Speech Assessment," IEEE ICME 2024, July 2024, (Top Performance on the Track 3 - VoiceMOS Challenge 2023)
