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
