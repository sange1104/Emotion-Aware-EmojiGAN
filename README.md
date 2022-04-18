# Emo2GAN
This repository provides the Tensorflow implementation code for the paper **Learning to Generate Expressive Emojis with Emotion**. This work generates express emojis from quantified emotion scores, using a CGAN-based model.

<!-- ![case_study_example_total_4](https://user-images.githubusercontent.com/63252403/163818058-b14319c4-fb26-420f-b919-7b62ea4d1fd3.jpg) -->
<img src="https://user-images.githubusercontent.com/63252403/163818058-b14319c4-fb26-420f-b919-7b62ea4d1fd3.jpg" width="500" height="130"/>

Overview😎
-------------
* [data/]() contains the directories for emoji corporates consisting of published emojis. Due to the license issue, you should place the dataset by yourself. Emoji dataset can be obtained from [here](https://www.kaggle.com/datasets/subinium/emojiimage-dataset). In addition, [EmoTag1200 file](https://github.com/abushoeb/EmoTag/blob/master/data/EmoTag1200-scores.csv) should be placed in this directory which provides the corresponding emotion scores for each emoji. The data directory is expected to consist as follows.
```bash
data
├── Apple
│   ├── emoji_1.jpg
│   ├── emoji_2.jpg
|   ├── ...
│   └── emoji_n.jpg
├── Samsung 
├── Google
├── ...
└── EmoTag1200-scores.csv 
```
* [scripts/]() contains code for implementing the model in the paper.
* [checkpoints/]() is a repository where a checkpoint of the trained model such as weight information or optimizer state would be saved.
* [outputs/]() is a repository where the generated images of emojis are saved.
* README.md
* requirements.txt

Environment setup
-------------
For experimental setup, ``requirements.txt`` lists down the requirements for running the code on the repository. Note that a cuda device is required.
The requirements can be downloaded using,
```
pip install -r requirements.txt
``` 

Our proposed Framework
-------------
<!-- ![model_architecture_v3](https://user-images.githubusercontent.com/63252403/163816911-515d32e6-0d24-48d3-92d2-f30b16beee7a.png) -->

<img src="https://user-images.githubusercontent.com/63252403/163816911-515d32e6-0d24-48d3-92d2-f30b16beee7a.png" width="600" height="250"/>

How to train
-------------
You can run [train.py]() setting arguments as follows:
 
* gpu_num: required, int, no default
* iters: not required, int, 500
* lambda_1: not required, float, 0.9
* lambda_2: not required, float, 0.99
* latent_dim: not required, int, 100
* lr_g: not required, float, 1e-04
* iters: not required, float, 1e-04
* batch_size: not required, int, 32 

<!-- |Name|Required|Type|Default|
|---|---|---|---|
|gpu_num|Yes|int|-|
|iters|No|int|500|
|lambda_1|No|float|0.9|
|lambda_2|No|float|0.99|
|latent_dim|No|int|100|
|lr_g|No|float|1e-04|
|lr_d|No|float|1e-04| 
|batch_size|No|int|32|  -->


You can train the model as follows:
```
python ./scripts/train.py --gpu_num 0 --iters 500
```  


Simple Demo
-------------
You can generate any emotional emoji you want to express as follows:
```
python ./scripts/demo.py --gpu_num 0
``` 

**Interface**

![demo](https://user-images.githubusercontent.com/63252403/163826964-7404af60-a578-4e7d-a900-04239d8c9921.JPG)

**Output image**

<img src="https://user-images.githubusercontent.com/63252403/163827069-2b6e03a8-6460-405e-bca1-71fb2a72de3f.png" width="45" height="35"/>

