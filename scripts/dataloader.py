import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2

#############################
# Data
#############################
def get_emoji_name(img_name):
    img_name = img_name.replace('.', '_')
    split = img_name.split('_')
    
    # remove corp information
    split = split[1:] 

    if 'augment' in split:
        # remove augment information
        aug_idx = split.index('augment')
        split = split[:aug_idx]
    else:
        # remove file format information
        aug_idx = split.index('jpg')
        split = split[:aug_idx]
    emoji_name = ' '.join(split)
    return emoji_name

def get_score(img_name, df_emotag):
    # get emoji name from image file 
    emoji_name = get_emoji_name(img_name)
    # get row for certain emoji name
    row = df_emotag[df_emotag.name==emoji_name]
    # get score vector
    assert df_emotag.columns[3] == 'anger'
    score = np.array(row.iloc[:,3:]) 
    return score

def get_norm_img(path, img_size):
    img_arr = np.array(Image.open(path)) 
    # resize
    img_arr = cv2.resize(img_arr, (img_size, img_size))
    # normalize
    img_arr = (img_arr - 127.5) / 127.5
    return img_arr

def get_data(data_dir, df_path, img_size, shuffle=True):
    '''
    Arguments
     - data_dir: top directory consists of emotion directories
     - df_path: path of emo_tag csv file
    Returns
     - image_list: a list consists of image array
     - emo_emb_list: a list consists of score vector for each image
    ''' 
    df_emotag = pd.read_csv(df_path)
    
    # initialize 
    image_list = []
    emo_emb_list = []
    total_num = 0
    
    # check if data directory is categorized ver or not
    type = data_dir.split('/')[-1]
    if type == 'categorized':
        # get emotion list
        emotion_list = os.listdir(data_dir)
        emo_to_label = {j:i for i,j in enumerate(emotion_list)} 

        for emo in emotion_list:
            print('Load %s emoji...'%emo)
            emo_dir = os.path.join(data_dir, emo)
            emo_num = 0

            for corps in os.listdir(emo_dir):
                if corps in ['Gmail', 'SoftBank', 'DoCoMo', 'KDDI', 'Samsung']:
                    continue

                corps_dir = os.path.join(emo_dir, corps)  
                for img in os.listdir(corps_dir):
                    # 1) get image array
                    path = os.path.join(corps_dir, img)
                    img_arr = get_norm_img(path, img_size) 
                    image_list.append(img_arr)

                    # 2) get emotion condition vector 
                    emb = get_score(img, df_emotag)
                    emo_emb_list.append(emb)

                # cumulate number
                num = len(os.listdir(corps_dir))
                emo_num += num

            total_num += emo_num
            print('Get %d data!'%emo_num)
            print()

        image_arr = np.stack(image_list)
        emo_emb_arr = np.concatenate(emo_emb_list, axis=0)
        print('-'*50)
        print('Total number of data is %d'%total_num)
    
    elif type == 'uncategorized': 
        for corps in os.listdir(data_dir):
            if corps in ['Gmail', 'SoftBank', 'DoCoMo', 'KDDI', 'Samsung']:
                continue 
            print('Load %s emoji...'%corps)

            corps_dir = os.path.join(data_dir, corps)  
            for img in os.listdir(corps_dir):
                # 1) get image array
                path = os.path.join(corps_dir, img)
                img_arr = get_norm_img(path, img_size) 
                image_list.append(img_arr)

                # 2) get emotion condition vector
#                 print(img)
                emb = get_score(img, df_emotag) 
                emo_emb_list.append(emb)

            # cumulate number
            corp_num = len(os.listdir(corps_dir)) 

            total_num += corp_num
            print('Get %d data!'%corp_num)
            print()

        image_arr = np.stack(image_list)
        emo_emb_arr = np.concatenate(emo_emb_list, axis=0) 
        print('-'*50)
        print('Total number of data is %d'%total_num)

    if shuffle:
        # shuffle dataset
        shuffle_idx = list(range(total_num))
        np.random.shuffle(shuffle_idx)

        image_arr = image_arr[shuffle_idx]
        emo_emb_arr = emo_emb_arr[shuffle_idx]
    
    return image_arr, emo_emb_arr