# source code from https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

from tqdm import tqdm
import tensorflow as tf
import numpy as np
import math
from scipy import linalg

def compute_embeddings(dataloader, count):
    image_embeddings = []
    for _ in tqdm(range(count)):
        images = next(iter(dataloader))  
        embeddings = inception_model.predict(images) 
        image_embeddings.extend(embeddings) 
    return np.array(image_embeddings)

def calculate_fid(trainset, generated_images): 
    batch_size = 100 
    count = math.ceil(10000/batch_size)
    # make dataloader 
    trainloader = trainset.batch(batch_size)
    # compute embeddings for real images
    real_embeddings = compute_embeddings(trainloader, count) 
    
    # make dataloader 
    gendataset = tf.data.Dataset.from_tensor_slices(generated_images)
    genloader = gendataset.batch(batch_size)
    # compute embeddings for generated images
    generated_embeddings = compute_embeddings(genloader, count)
    

    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings, rowvar=False)
    
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
 
# get pre-trained inception V3 model
inception_model = tf.keras.applications.InceptionV3(include_top=False, 
                              weights="imagenet", 
                              pooling='avg')