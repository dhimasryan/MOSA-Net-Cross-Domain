# Deep Learning-based Non-Intrusive Multi-Objective Speech Assessment Model with Cross-Domain Features

### Introduction ###

The non-intrusive speech assessment metrics have garnered significant attention in recent years, and several deep learning-based models have been developed accordingly. 
Although these models are more flexible than conventional speech assessment metrics, most of them are designed to estimate a specific evaluation score, whereas speech assessment generally involves multiple facets. Herein, we propose a cross-domain multi-objective speech assessment model called MOSA-Net, which can estimate multiple speech assessment metrics simultaneously. More specifically, MOSA-Net is designed to estimate the speech quality, intelligibility, and distortion assessment scores of an input test speech signal. It comprises a convolutional neural network and bidirectional long short-term memory (CNN-BLSTM) architecture for representation extraction, and a multiplicative attention layer and a fully-connected layer for each assessment metric. In addition, cross-domain features (spectral and time-domain features) and latent representations from self-supervised learned models are used as inputs to combine rich acoustic information from different speech representations to obtain more accurate assessments. Experimental results show that MOSA-Net can precisely predict perceptual evaluation of speech quality (PESQ), short-time objective intelligibility (STOI), and speech distortion index (SDI) scores when tested on noisy and enhanced speech utterances under either seen test conditions or unseen test conditions. Moreover, MOSA-Net, originally trained to assess objective scores, can be used as a pre-trained model to be effectively adapted to an assessment model for predicting subjective quality and intelligibility scores with a limited amount of training data. In light of the confirmed prediction capability, we further adopt the latent representations of MOSA-Net to guide the speech enhancement (SE) process and derive a quality-intelligibility (QI)-aware SE (QIA-SE) approach accordingly. Experimental results show that QIA-SE provides superior enhancement performance compared with the baseline SE system in terms of objective evaluation metrics and qualitative evaluation test.

For more detail please check our <a href="https://arxiv.org/pdf/2111.02363v2.pdf" target="_blank">Paper</a>

### Installation ###

You can download our environmental setup at Environment Folder and use the following script.
```js
conda env create -f environment.yml
```

Please be noted, that the above environment is specifically used to run ```MOSA-Net_Cross_Domain.py, Generate_PS_Feature.py, Generate_end2end_Feature.py```. To generate Self Supervised Learning (SSL) feature, please use ```python 3.6``` and follow the instructions in following <a href="https://github.com/pytorch/fairseq" target="_blank">link</a> to deploy fairseq module.  
### Feature Extaction ###

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
R. E. Zezario, S.-W. Fu, F. Chen, C.-S. Fuh, Y. Tsao, and H.-M. Wang, “Deep Learning-based Non-Intrusive Multi-Objective Speech Assessment Model with Cross-Domain Features,” in arXiv:2111.02363, 2021

### Note ###

<a href="https://github.com/CyberZHG/keras-self-attention" target="_blank">Self Attention</a>, <a href="https://github.com/grausof/keras-sincnet" target="_blank">SincNet</a>, <a href="https://github.com/pytorch/fairseq" target="_blank">Self-Supervised Learning Model</a> are created by others
