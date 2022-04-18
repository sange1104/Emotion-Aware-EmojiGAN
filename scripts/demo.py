'''
Generate samples when giving specified emotion vectors.
output: npy file consisting of image array
'''
import os
import numpy as np
import argparse
from tensorflow.keras.layers import Input 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam  
from PIL import Image

from model import define_discriminator, define_generator
from sampling import generate_latent_points

def load_pretrained_model(epoch):
    d_model = define_discriminator(embedding_dim, image_shape) 
    g_model = define_generator(latent_dim, embedding_dim) 
    opt = Adam(lr=0.0001, beta_1=0.5)
    d_model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

    # define gan
    z = Input(shape=(latent_dim,)) 
    cond_input = Input(shape=(embedding_dim,))
    img = g_model([z, cond_input])

    # For the combined model we will only train the generator
    d_model.trainable = False

    # The discriminator takes generated images as input and determines validity
    valid = d_model([img, cond_input]) 
    combined = Model([z, cond_input], valid)

    # compile model
    opt = Adam(lr=0.0001, beta_1=0.5)
    combined.compile(loss='binary_crossentropy', optimizer=opt)

    # load trained parameters
    ckp_path = './checkpoints/emo2gan_epoch_%d.h5'%(epoch+1)
    g_model.load_weights(ckp_path)
    print('Done loading models!')
    return g_model 

#############################
# Main
#############################
if __name__=='__main__':
    # configuration
    parser = argparse.ArgumentParser()  
    parser.add_argument('--gpu_num', type=int, required=True)  
    args = parser.parse_args()  
    gpu_num = args.gpu_num 
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%gpu_num 
    embedding_dim = 8
    image_shape = (80, 80, 3)
    latent_dim = 248
    epoch = 260

    print('-'*120)
    print('A simple demo for emoji generation with your own emotions!')
    print('Please type the emotion score in the following order, then our model will generate emojis reflecting your emotion.')
    print('-'*120)
    n_samples = int(input('Input number of samples: '))
    x_input = generate_latent_points(latent_dim, n_samples) 
    emotion_list = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    cond_list = []
    for i in range(n_samples):
        cond_list.append([])
        for emo in emotion_list:
            cond = float(input('%s : '%emo)) 
            cond_list[i].append(cond) 
        print()
    cond_list = np.array(cond_list) 

    # load model
    g_model = load_pretrained_model(epoch)

    # generate images
    X_gen = g_model.predict([x_input, cond_list])
    print('Done generation!')

    # save image
    output_dir = './outputs'
    for i in range(n_samples):
        # preprocess
        img = (X_gen[i] + 1) / 2
        img = np.array(img*255).astype(np.uint8)
        img = Image.fromarray(img)
        # save
        name = [e+'_'+str(int(c)) for e,c in zip(emotion_list, cond_list[i])]
        save_name = 'emojis_with_%s_%s_%s_%s_%s_%s_%s_%s.png'%tuple(name)
        save_path = os.path.join(output_dir, save_name)
        img.save(save_path)
        