# Deep Learning-based Non-Intrusive Multi-Objective Speech Assessment Model with Cross-Domain Features

### Introduction ###

This study proposes a cross-domain multi-objective speech assessment model, called MOSA-Net, which can simultaneously estimate the speech quality, intelligibility, and distortion assessment scores of an input speech signal. MOSA-Net comprises a convolutional neural network and bidirectional long short-term memory architecture for representation extraction, and a multiplicative attention layer and a fully connected layer for each assessment metric prediction. Additionally, cross-domain features (spectral and time-domain features) and latent representations from self-supervised learned (SSL) models are used as inputs to combine rich acoustic information to obtain more accurate assessments. Experimental results show that in both seen and unseen noise environments, MOSA-Net can improve the linear correlation coefficient (LCC) scores in perceptual evaluation of speech quality (PESQ) prediction, compared to Quality-Net, an existing single-task model for PESQ prediction, and improve LCC scores in short-time objective intelligibility (STOI) prediction, compared to STOI-Net, an existing single-task model for STOI prediction. Moreover, MOSA-Net can be used as a pre-trained model to be effectively adapted to an assessment model for predicting subjective quality and intelligibility scores with a limited amount of training data. Experimental results show that MOSA-Net can improve LCC scores in mean opinion score (MOS) predictions, compared to MOS-SSL, a strong single-task model for MOS prediction.

For more detail please check our <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9905733" target="_blank">Paper</a>

The implementation of MOSA-Net with Whisper features (MOSA-Net+) can be checked in the following folder <a href="https://github.com/dhimasryan/MOSA-Net-Cross-Domain/tree/main/MOSA_Net%2B" target="_blank">MOSA-Net-Cross-Domain/MOSA_Net+</a> 

### Installation ###

You can download our environmental setup at Environment Folder and use the following script.
```js
conda env create -f environment.yml
```

Please be noted, that the above environment is specifically used to run ```MOSA-Net_Cross_Domain.py, Generate_PS_Feature.py, Generate_end2end_Feature.py```. To generate Self Supervised Learning (SSL) feature, please use ```python 3.6``` and follow the instructions in following <a href="https://github.com/pytorch/fairseq" target="_blank">link</a> to deploy fairseq module.  
### Feature Extraction ###

For extracting cross-domain features, please use Generate_end2end_Feature.py, Generate_PS_Feature.py, Generate_SSL_Feature.py. When extracting SSL feature, please make sure that fairseq can be imported correctly. Please refer to this link for detail <a href="https://github.com/pytorch/fairseq" target="_blank">installation</a>. 

Please follow the following format to make the input list.
```js
PESQ score, STOI score, SDI score, filepath directory
```

### How to run the code ###

Please use following script to train the model:
```js
python MOSA-Net_Cross_Domain.py --gpus <assigned GPU> --mode train
```
For, the testing stage, plase use the following script:
```js
python MOSA-Net_Cross_Domain.py --gpus <assigned GPU> --mode test
```

### Citation ###

Please kindly cite our paper, if you find this code is useful.

<a id="1"></a> 
R. E. Zezario, S. -W. Fu, F. Chen, C. -S. Fuh, H. -M. Wang and Y. Tsao, "Deep Learning-Based Non-Intrusive Multi-Objective Speech Assessment Model With Cross-Domain Features," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 54-70, 2023, doi: 10.1109/TASLP.2022.3205757.

### Note ###

<a href="https://github.com/CyberZHG/keras-self-attention" target="_blank">Self Attention</a>, <a href="https://github.com/grausof/keras-sincnet" target="_blank">SincNet</a>, <a href="https://github.com/pytorch/fairseq" target="_blank">Self-Supervised Learning Model</a> are created by others
