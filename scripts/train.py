
import argparse
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from model import define_discriminator, define_generator, l1_loss, mean_square_error
from dataloader import get_data
from sampling import generate_fake_samples, generate_latent_points
from utils import *

def train(g_model, d_model, gan_model, dataset, emo_embedding, latent_dim, history_path=None, n_epochs=100, n_batch=64, save=True):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    y_real = np.ones((n_batch, 1))
    y_fake = np.zeros((n_batch, 1))
    
    # manually enumerate epochs
    for i in range(n_epochs):
        d_loss = []
        g_loss = []
        # enumerate batches over the training set
        for j in tqdm(range(bat_per_epo)):
            # get 'real' samples
            X_real = dataset[j*n_batch:(j+1)*n_batch] 
            condition = emo_embedding[j*n_batch:(j+1)*n_batch]
            
            # update discriminator model weights
            d_model.trainable=True  
            _, adv_loss_t, emo_loss_t, acc_real_t = d_model.train_on_batch(X_real, [y_real, condition]) 
            d_loss.append([adv_loss_t, emo_loss_t])
            
            # generate 'fake' examples
            X_fake, _ = generate_fake_samples(g_model, condition, latent_dim, n_batch)
            # update discriminator model weights
            _, adv_loss_f, emo_loss_f, acc_real_f= d_model.train_on_batch(X_fake, [y_fake, condition]) 
            d_loss.append([adv_loss_f, emo_loss_f])

            # prepare points in latent space as input for the generator 
            X_gan = generate_latent_points(latent_dim, n_batch*2) 
            # update the generator via the discriminator's error  
            double_condition = np.concatenate([condition, condition])

            d_model.trainable=False  
            # print(gan_model.train_on_batch([X_gan, double_condition], [np.ones((n_batch*2, 1)), np.concatenate([X_real, X_real]), double_condition], return_dict))
            _, adv_loss_g, rec_loss_g, emo_loss_g = gan_model.train_on_batch([X_gan, double_condition], [np.ones((n_batch*2, 1)), np.concatenate([X_real, X_real]), double_condition])
            g_loss.append([adv_loss_g, rec_loss_g, emo_loss_g])
            # summarize loss on this batch
#             print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
# #                 (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss)) 
        if (i+1) % 5 == 0 and save:
            summarize_performance(i, g_model, d_model, dataset, emo_embedding, latent_dim)
        
        if history_path != None:
            # save history txt file 
            d_loss = np.mean(np.array(d_loss), axis=0)
            g_loss = np.mean(np.array(g_loss), axis=0)
            with open(history_path, "a") as f:
                f.write("[Epoch %d] Adv D Loss: %.4f Emo D Loss: %.4f || Adv G Loss: %.4f Rec G loss: %.4f Emo G Loss: %.4f"%(i, d_loss[0], d_loss[1], g_loss[0], g_loss[1], g_loss[2])) 
                f.write("\n")

#############################
# Main
#############################
if __name__=='__main__':
    # configuration
    parser = argparse.ArgumentParser()  
    parser.add_argument('--gpu_num', type=int, required=True) 
    parser.add_argument('--iters', type=int, required=False, default=500) 
    parser.add_argument('--lambda_1', type=float, required=False, default=0.9) 
    parser.add_argument('--lambda_2', type=float, required=False, default=0.99)
    parser.add_argument('--latent_dim', type=int, required=False, default=100)
    parser.add_argument('--lr_g', type=float, required=False, default=0.0001) 
    parser.add_argument('--lr_d', type=float, required=False, default=0.0001)
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    args = parser.parse_args()  
    gpu_num = args.gpu_num 
    epoch = args.iters 
    lambda_1 = args.lambda_1 
    lambda_2 = args.lambda_2  
    latent_dim = args.latent_dim  
    lr_g = args.lr_g 
    lr_d = args.lr_d  
    batch_size = args.batch_size 

    # set gpu device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%gpu_num 
 
    data_dir = '../data'  
    file_name = '../data/EmoTag1200-scores.csv'  

    image_shape = (80, 80, 3) 
    assert image_shape[0] == image_shape[1]
    embedding_dim = 8  

    img_size = image_shape[0]
    images, conds = get_data(data_dir, file_name, img_size) 
    
    d_model = define_discriminator(embedding_dim, image_shape) 
    g_model = define_generator(latent_dim, embedding_dim) 
    opt_d = Adam(lr=lr_d, beta_1=0.5) 
    losses_d = {
        "binary_output": "binary_crossentropy",
        "emotion_output": mean_square_error,
    }
    metrics = {
        "binary_output": "accuracy" 
    }
    loss_weights_d = {"binary_output": lambda_1, "emotion_output": 1.0 - lambda_1}
    d_model.compile(loss=losses_d, optimizer=opt_d, metrics=metrics, loss_weights=loss_weights_d)

    # define gan
    z = Input(shape=(latent_dim,)) 
    cond_input = Input(shape=(embedding_dim,))
    img = g_model([z, cond_input])

    # For the combined model we will only train the generator
    d_model.trainable = False

    # The discriminator takes generated images as input and determines validity
    valid, cond_output = d_model(img) 
    combined = Model([z, cond_input], [valid, img, cond_output], name='Combined')   

    # compile model
    opt_g = Adam(lr=lr_d, beta_1=0.5) 
    losses_g = {
        "D": "binary_crossentropy",
        "G": l1_loss,
        "D_1": mean_square_error,
    }
    loss_weights_g = {"D": lambda_1 * lambda_2, "G": (1.0 - lambda_1) * lambda_2, "D_1": (1.0 - lambda_2)}

    combined.compile(loss=losses_g, optimizer=opt_g, loss_weights=loss_weights_g) 

    history_dir = '../history'
    history_path = os.path.join(history_dir, "CGAN_v7.txt")
    with open(history_path, "w") as f:
        f.write("History") 
        f.write('\n')
    train(g_model, d_model, combined, images, conds, latent_dim, history_path, epoch, batch_size, True) 