# MOSA-Net+ : An improved version of MOSA-Net (TF version)

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

R. E. Zezario, Y.-W. Chen, S.-W. Fu, Y. Tsao, H.-M. Wang, C.-S. Fuh, "A Study on Incorporating Whisper for Robust Speech Assessment," IEEE ICME 2024, July 2024, (Top Performance on the Track 3 - VoiceMOS Challenge 2023)
