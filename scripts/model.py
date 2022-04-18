import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2DTranspose, LeakyReLU, Dense, Reshape, Flatten, Dropout, concatenate, BatchNormalization, Activation, ZeroPadding2D, Conv2D
from tensorflow.keras.models import Sequential, Model

#############################
# Model
#############################

def define_discriminator(embedding_dim, img_shape): 
    # Weight initialization
    initializer = tf.keras.initializers.Orthogonal()

    discriminator_input = Input(shape=img_shape, name="d_input")
    
    # img
    D = Conv2D(128, (5, 5), padding="same", kernel_initializer=initializer)(discriminator_input)
    D = LeakyReLU(alpha=0.2)(D)
    
    D = Conv2D(128, (5, 5), strides=(2, 2), padding="same", kernel_initializer=initializer)(D) 
    # D = BatchNormalization(momentum=0.8)(D)
    D = LeakyReLU(alpha=0.2)(D)
    
    D = Conv2D(128, (5, 5), strides=(2, 2), padding="same", kernel_initializer=initializer)(D) 
    # D = BatchNormalization(momentum=0.8)(D)
    D = LeakyReLU(alpha=0.2)(D)
    
    D = Conv2D(128, (5, 5), strides=(2, 2), padding="same", kernel_initializer=initializer)(D) 
    # D = BatchNormalization(momentum=0.8)(D)
    D = LeakyReLU(alpha=0.2)(D)
    
    D = Conv2D(128, (5, 5), strides=(2, 2), padding="same", kernel_initializer=initializer)(D) 
    # D = BatchNormalization(momentum=0.8)(D)
    D = LeakyReLU(alpha=0.2)(D)
    
    D = Conv2D(128, (5, 5), strides=(2, 2), padding="same", kernel_initializer=initializer)(D) 
    # D = BatchNormalization(momentum=0.8)(D)
    D = LeakyReLU(alpha=0.2)(D)

    D = Flatten()(D)
    D = Dropout(0.4)(D) 
    y_1 = Dense(1, activation="sigmoid", kernel_initializer=initializer, name='binary_output')(D) 
    y_2 = Dense(8, kernel_initializer=initializer, name='emotion_output')(D) 

    discriminator = Model([discriminator_input], [y_1, y_2], name='D')  
    return discriminator

def define_generator(latent_dim, embedding_dim, channels=3):
    # Weight initialization
    initializer = tf.keras.initializers.Orthogonal()

    generator_input = Input(shape=(latent_dim,), name="g_input") # (100, )
    cond_input = Input(shape=(embedding_dim,), name="cond_g_input") # (8, )

    # cond 
    cond_output = Dense(latent_dim, kernel_initializer=initializer)(cond_input) # (256, )
    cond_output = Dropout(0.4)(cond_output) 

    # multiply
    # mul = tf.math.multiply(generator_input, cond_output)
    
    # concat
    mul = concatenate([generator_input, cond_output]) # (batch_size, latent_dim*2)
    G = Dense(128 * 5 * 5, kernel_initializer=initializer)(mul)
    G = LeakyReLU(alpha=0.2)(G)
    G = Reshape((5, 5, 128))(G)
    
    G = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(G)
    # G = BatchNormalization(momentum=0.8)(G)
    G = LeakyReLU(alpha=0.2)(G)
    
    G = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(G)
    # G = BatchNormalization(momentum=0.8)(G)
    G = LeakyReLU(alpha=0.2)(G)
    
    G = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(G)
    # G = BatchNormalization(momentum=0.8)(G)
    G = LeakyReLU(alpha=0.2)(G)
    
    G = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer)(G)
    # G = BatchNormalization(momentum=0.8)(G)
    G = LeakyReLU(alpha=0.2)(G)
    
    G = Conv2D(filters=3, kernel_size=(5, 5), padding="same",  activation="tanh", kernel_initializer=initializer, name="image_output")(G)
    generator = Model([generator_input, cond_input], G, name='G')
    return generator 

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0003, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

#############################
# Loss Function
#############################
def l1_loss(y_true, y_pred):
    return tf.math.reduce_sum(tf.math.abs(y_true-y_pred), axis=-1)
    
def mean_square_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.math.square(y_true-y_pred), axis=-1)