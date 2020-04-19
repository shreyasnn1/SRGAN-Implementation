import numpy as numpy
import tensorflow as tf
import keras
import cv2
from tqdm import tqdm
from utils import *
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add

def add_res_block(model,filters = 64,kernel_size = 3, strides = 1):
    model_tmp = model

    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = PReLU(shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)

    model = add([model_tmp, model])

    return model

def pixelShuffler_block(model, kernel_size = 3, filters = 256, strides = 1):

    #model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = UpSampling2D(size = 2)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model

def discriminator_block(model, filters, kernel_size, strides):
    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model
    
class SRGAN():
    def __init__(self):
        
        self.image_shape = (128,128,3)
        self.downscale_factor = 4
        self.noise_shape = (self.image_shape[0]//self.downscale_factor, self.image_shape[1]//self.downscale_factor, self.image_shape[2])


        self.generator = self.create_gen()
        self.discriminator = self.create_dis()

        # self.discriminator.compile(optimizer = get_optimizer(), loss = VGG_LOSS(self.image_shape))
        # self.gan_model.compile(optimizer = get_optimizer(), "binary_crossentropy")
        optimizer = get_optimizer()
        self.vgg_loss = VGG_LOSS(self.image_shape).vgg_loss
        self.generator.compile(loss=self.vgg_loss, optimizer=optimizer)
        self.discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)




    def create_gen(self):

        gen_input = Input(shape = self.noise_shape)
	    
        model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(gen_input)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
	    
        model_tmp = model
        
        # Using 16 Residual Blocks
        for index in range(16):
	        model = add_res_block(model)
	    
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
        model = BatchNormalization(momentum = 0.5)(model)
        model = add([model_tmp, model])
	    
	    # Using 2 UpSampling Blocks
        for index in range(2):
            model = pixelShuffler_block(model, 3, 256, 1)

        model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model)
        model = Activation('tanh')(model)

        generator_model = Model(inputs = gen_input, outputs = model)

        print('Generator Model: ')
        print(generator_model.summary())
        return generator_model
    
    def create_dis(self):

        disc_input = Input(shape = self.image_shape)

        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(disc_input)
        model = LeakyReLU(alpha=0.2)(model)

        model = discriminator_block(model, filters = 64, kernel_size = 3, strides = 2)
        model = discriminator_block(model, filters = 128, kernel_size = 3, strides = 1)
        model = discriminator_block(model, filters = 128, kernel_size = 3, strides = 2)
        model = discriminator_block(model, filters = 256, kernel_size = 3, strides = 1)
        model = discriminator_block(model, filters = 256, kernel_size = 3, strides = 2)
        model = discriminator_block(model, filters = 512, kernel_size = 3, strides = 1)
        model = discriminator_block(model, filters = 512, kernel_size = 3, strides = 2)
        model = Flatten()(model)
        model = Dense(256)(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = Dense(1)(model)
        model = Activation('sigmoid')(model)


        discriminator_model = Model(inputs = disc_input, outputs = model)
        print('Discriminator Model: ')
        print(discriminator_model.summary())
        return discriminator_model

    def get_gan_network(self, vgg_loss):


        make_trainable(self.discriminator, False)
        gan_input = Input(shape=self.noise_shape)
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)
        print("==========Creating Model============")
        gan = Model(inputs=gan_input, outputs=[x,gan_output])
        gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                    loss_weights=[1., 1e-3],
                    optimizer=get_optimizer())

        return gan
    


    def train(self, epochs, batch_size,model_save_dir, input_path, output_dir):
        train_hr, train_lr,test_hr, test_lr = load_images(input_path)
        print("Loaded Images!!!!!!")

        
        batch_count = int(train_hr.shape[0] / batch_size)

        
        print("Starting get_gan_network function")
        gan = self.get_gan_network(self.vgg_loss)
        
        loss_file = open(model_save_dir + 'losses.txt' , 'w+')
        loss_file.close()

        for epoch in range(epochs):
            print ('-'*15, 'Epoch ', epoch+1, '-'*15)
            for _ in tqdm(range(batch_count)):
                
                indices = np.random.randint(0, train_hr.shape[0], size=batch_size)
                
                batch_hr = train_hr[indices]
                batch_lr = train_lr[indices]
                generated_images_sr = self.generator.predict(batch_lr)
                

                true_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
                fake_Y = np.random.random_sample(batch_size)*0.2
                
                make_trainable(self.discriminator, True)
                
                d_loss_real = self.discriminator.train_on_batch(batch_hr, true_Y)
                d_loss_fake = self.discriminator.train_on_batch(generated_images_sr, fake_Y)
                discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                
                indices = np.random.randint(0, train_hr.shape[0], size=batch_size)
                batch_hr = train_hr[indices]
                batch_lr = train_lr[indices]

                gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
                make_trainable(self.discriminator, False)
                gan_loss = gan.train_on_batch(batch_lr, [batch_hr,gan_Y])
                
                
            print("discriminator_loss : %f" % discriminator_loss)
            print("gan_loss :", gan_loss)
            gan_loss = str(gan_loss)
            
            loss_file = open(model_save_dir + 'losses.txt' , 'a')
            loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' %(epoch, gan_loss, discriminator_loss) )
            loss_file.close()

            if epoch == 1 or epoch % 5 == 0:
                plot_sr(self.generator, test_hr, test_lr)
            if epoch % 100 == 0:
                self.generator.save(model_save_dir + 'gen_model%d.h5' % epoch)
                self.discriminator.save(model_save_dir + 'dis_model%d.h5' % epoch)










