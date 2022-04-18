import matplotlib.pyplot as plt
import numpy as np

from sampling import generate_real_samples, generate_fake_samples

# create and save a plot of generated images
def save_plot(examples, epoch, n=10):
    # scale from [-1,1] to [0,1]
    k = 8
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(k * n):
        # define subplot
        plt.subplot(k, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i])
    # save plot to file
    # filename = './output/CGAN_v3/v3_a_%.2f_generated_plot_e%03d.png' % (ALPHA, epoch+1)
    filename = './outputs/generated_epoch_%03d.png' % (epoch+1)
    plt.savefig(filename)
    plt.close()

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, emo_embedding, latent_dim, n_samples=80):
    # prepare real samples
    X_real, cond_real, y_real = generate_real_samples(dataset, emo_embedding, n_samples)
    # evaluate discriminator on real examples  
    loss, binary_output_loss, emotion_output_loss, acc_real  = d_model.evaluate(X_real, [y_real, cond_real], verbose=0) 
    # prepare fake examples 
    cond_real = np.zeros((80, 8))
    for i in range(8):
        cond_real[i*10:(i+1)*10, i] = 1
    x_fake, _ = generate_fake_samples(g_model, cond_real, latent_dim, n_samples)
    y_fake = np.zeros((n_samples, 1))
    # evaluate discriminator on fake examples
    loss, binary_output_loss, emotion_output_loss, acc_fake = d_model.evaluate(x_fake,  [y_real, cond_real], verbose=0)
#     print(d_model.predict([x_fake, cond_real]))
    # summarize discriminator performance
    print('[Epoch %d] Accuracy real: %.0f%%, fake: %.0f%%' % (epoch+1, acc_real*100, acc_fake*100))
    # save plot
    save_plot(x_fake, epoch)
    # save the generator model tile file
    
    # filename = './checkpoints/v3_a_%.2f_CGAN_generator_epoch_%d.h5'%(ALPHA, epoch+1)
    filename = './checkpoints/emo2gan_epoch_%d.h5'%(epoch+1)
    g_model.save(filename)
 
def denormalize(arr):
    return ((arr*127.5)+127.5).astype(np.uint8)