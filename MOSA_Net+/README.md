# MOSA-Net+ : An improved version of MOSA-Net 

This section of this repository aims to introduce an improved version of MOSA-Net model, namely MOSA-Net +, by leveraging the acoustic features from <a href="https://github.com/openai/whisper" target="_blank">Whisper</a>. The detail of the model architecture can be found in our <a href="https://arxiv.org/pdf/2309.12766.pdf" target="_blank">here.</a> 

### Installation ###

You can download our environmental setup at MOSA-Net-Cross-Domain/Environment Folder and use the following script.
```js
conda env create -f environment.yml
```

Please be noted, that the above environment is specifically used to run ```MOSANet_plus_VoiceMOS2023.py```. To generate Whisper feature, please follow the environment instructions in the following <a href="https://github.com/openai/whisper" target="_blank">link</a>

### How to run the code ###

To extract the Whisper features, please use the following command.
```js
python Generate_Whisper_Feature.py
```
Please use following script to train the model:
```js
python MOSANet_plus_VoiceMOS2023.py --gpus <assigned GPU> --mode train
```
For, the testing stage, plase use the following script:
```js
python MOSANet_plus_VoiceMOS2023.py --gpus <assigned GPU> --mode test
```

### Citation ###

Please kindly cite our paper, if you find this code is useful.

<a id="1"></a> 
R. E. Zezario, S. -W. Fu, F. Chen, C. -S. Fuh, H. -M. Wang and Y. Tsao, "Deep Learning-Based Non-Intrusive Multi-Objective Speech Assessment Model With Cross-Domain Features," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 54-70, 2023, doi: 10.1109/TASLP.2022.3205757.
