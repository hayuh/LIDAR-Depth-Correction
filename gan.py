#Inspired by: https://www.youtube.com/watch?v=Mng57Tj18pc&t=1837s
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.datasets import mnist
import psutil
import time

# example of calculating the frechet inception distance in Keras
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize #used 'python -m pip install scikit-image'

tic = time.perf_counter()

#Define input image dimensions
#Large images take too much time and resources.
img_rows = 720
img_cols = 1280
channels = 1
img_shape = (img_rows, img_cols, channels)


#Given input of noise (latent) vector, the Generator produces an image.
def build_generator():

    noise_shape = (100,) #1D array of size 100 (latent vector / noise)

#Define your generator network 
#Here we are only using Dense layers. But network can be complicated based
#on the application. For example, you can use VGG for super res. GAN.         

    model = Sequential()

    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    #model.add(Dense(1024))
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(np.prod(img_shape), activation='tanh')) #originally tanh, changed to linear because outputs are not -1 to 1.
    model.add(Reshape(img_shape))

    model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)    #Generated image
    return Model(noise, img)
#Alpha — α is a hyperparameter which controls the underlying value to which the
#function saturates negatives network inputs.
#Momentum — Speed up the training
##########################################################################

#Given an input image, the Discriminator outputs the likelihood of the image being real.
    #Binary classification - true or false (we're calling it validity)

def build_discriminator():


    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid')) #sigmoid activation function b/c output 0-1.
    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)
    return Model(img, validity)
#The validity is the Discriminator’s guess of input being real or not.

#Now that we have constructed our two models it’s time to pit them against each other.
#We do this by defining a training function, loading the data set, re-scaling our training
#images and setting the ground truths. 
#Epochs dictate the number of backward and forward propagations, the batch_size
#indicates the number of training samples per backward/forward propagation, and the
#sample_interval specifies after how many epochs we call our sample_image function
def train(X_train, epochs, batch_size, save_interval):

    # Load the dataset. Dim: 10x720x1280
    #X_train = load_data

    # Convert to float and Rescale -1 to 1 (Can also do 0 to 1)
    #X_train = (X_train.astype(np.float16) - 127.5) / 127.5

#Add channels dimension. As the input to our gen and discr. has a shape 28x28x1.
    X_train = np.expand_dims(X_train, axis=3) 

    half_batch = int(batch_size / 2)


#We then loop through a number of epochs to train our Discriminator by first selecting
#a random batch of images from our true dataset, generating a set of images from our
#Generator, feeding both set of images into our Discriminator, and finally setting the
#loss parameters for both the real and fake images, as well as the combined loss. 
    
    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half batch of real images
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx] #training set

 
        noise = np.random.normal(0, 1, (half_batch, 100))

        # Generate a half batch of fake images
        gen_imgs = generator.predict(noise) #validation set

        # Train the discriminator on real and fake images, separately
        #Research showed that separate training is more effective. 
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
    #take average loss from real and fake images. 
    #
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) 

#And within the same loop we train our Generator, by setting the input noise and
#ultimately training the Generator to have the Discriminator label its samples as valid
#by specifying the gradient loss.
        # ---------------------
        #  Train Generator
        # ---------------------
#Create noise vectors as input for generator. 
#Create as many noise vectors as defined by the batch size. 
#Based on normal distribution. Output will be of size (batch size, 100)
        noise = np.random.normal(0, 1, (batch_size, 100)) 

        # The generator wants the discriminator to label the generated samples
        # as valid (ones)
        #This is where the genrator is trying to trick discriminator into believing
        #the generated image is true (hence value of 1 for y)
        valid_y = np.array([1] * batch_size) #Creates an array of all ones of size=batch size

        # Generator is part of combined where it got directly linked with the discriminator
        # Train the generator with noise as x and 1 as y. 
        # Again, 1 as the output as it is adversarial and if generator did a great
        #job of folling the discriminator then the output would be 1 (true)
        g_loss = combined.train_on_batch(noise, valid_y) #STOPS HERE


#Additionally, in order for us to keep track of our training process, we print the
#progress and save the sample image output depending on the epoch interval specified.  
# Plot the progress
        
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        # If at save interval => save generated image samples
        if epoch % save_interval == 0:
            save_imgs(epoch)
        

#when the specific sample_interval is hit, we call the
#sample_image function. Which looks as follows.

def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 9000, which is range of Intel Realsense LIDAR camera in mm
    gen_imgs = (gen_imgs + 1)*4500 #shape: (25, 720, 1280, 1). 25 720x1280 arrays
    
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray') #gen_imgs[cnt, :,:,0] = (720, 1280)
            axs[i,j].axis('off')
            #np.save("epoch%d_gen_img%d" % (epoch,cnt), gen_imgs[cnt, :,:,0]) #saving gen_imgs in npy array
            cnt += 1
    fig.savefig("%d.png" % epoch)
    plt.close()
    
    #calculate FID
    idx = np.random.randint(0, load_data.shape[0], 25) #Randomly select 25 idx between 0 and 8 (len of load_data)
    fid = calculate_fid(load_data[idx], gen_imgs[:,:,:,0]) #input is 25 720x1280 arrays
    fidDict["epoch"].append(epoch)
    fidDict["fid"].append(fid)



def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        plt.imshow(image, cmap='gray')
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        plt.imshow(new_image, cmap='gray')
        # store
        images_list.append(new_image)
    return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(images1, images2):
    # prepare the inception v3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    # resize images
    images1 = scale_images(images1, (299,299,3)) #images1 orig Value range:0-9506. new value range:0-9432.5
    images2 = scale_images(images2, (299,299,3)) #images2 orig Value range:-0.06-0.06. new value range:-0.017-0.0189
    print('Scaled', images1.shape, images2.shape)
    # pre-process images
    images1 = preprocess_input(images1) #-1 to 1
    for image in images1:
        plt.imshow(image, cmap='gray')
    images2 = preprocess_input(images2) #-0.999 to -0.992
    for image in images2:
        plt.imshow(image, cmap='gray')
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

##############################################################################
#This function saves our images for us to view
load_data = np.zeros((8, 720, 1280))
load_data[0] = np.load('61.125\depth_image_1652108090807780171.npy')
load_data[1] = np.load('86.75\depth_image_1652109334245600312.npy')
load_data[2] = np.load('88.5\depth_image_1652108912547216057.npy')
load_data[3] = np.load('100\depth_image_1652107419965833808.npy')
load_data[4] = np.load('102.25\depth_image_1652109163666183263.npy')
load_data[5] = np.load('106\depth_image_1652108478517283367.npy')
load_data[6] = np.load('113.38\depth_image_1652107776730755283.npy')
load_data[7] = np.load('148\depth_image_1652108688105927426.npy')

#Code to save sample imgs
'''
fig, axs = plt.subplots(4,2)
cnt = 0
for i in range(4):
    for j in range(2):
        axs[i,j].imshow(load_data[cnt], cmap='gray') #gen_imgs[cnt, :,:,0] = (720, 1280)
        axs[i,j].axis('off')
        cnt += 1
fig.savefig("real_sample_img.png")
plt.close()
'''

#Let us also define our optimizer for easy use later on.
#That way if you change your mind, you can change it easily here
optimizer = Adam(0.0002, 0.5)  #Learning rate and momentum.

# Build and compile the discriminator first. 
#Generator will be trained as part of the combined model, later. 
#pick the loss function and the type of metric to keep track.                 
#Binary cross entropy as we are doing prediction and it is a better
#loss function compared to MSE or other. 

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

#build and compile our Discriminator, pick the loss function

#SInce we are only generating (faking) images, let us not track any metrics.
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

##This builds the Generator and defines the input noise. 
#In a GAN the Generator network takes noise z as an input to produce its images.  
z = Input(shape=(100,))   #Our random input to the generator
img = generator(z)

#This ensures that when we combine our networks we only train the Generator.
#While generator training we do not want discriminator weights to be adjusted. 
#This Doesn't affect the above descriminator training.     
discriminator.trainable = False  

#This specifies that our Discriminator will take the images generated by our Generator
#and true dataset and set its output to a parameter called valid, which will indicate
#whether the input is real or not.  
valid = discriminator(img)  #Validity check on the generated image


#Here we combined the models and also set our loss function and optimizer. 
#Again, we are only training the generator here. 
#The ultimate goal here is for the Generator to fool the Discriminator.  
# The combined model  (stacked generator and discriminator) takes
# noise as input => generates images => determines validity

combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

fidDict = {"epoch":[], "fid":[]}

train(X_train=load_data, epochs=201, batch_size=8, save_interval=20)

#Measuring time to train
toc = time.perf_counter()
print(f"Completed in {toc - tic:0.4f} seconds")

'''
# fid between real images
real_fid = []
for i in range(0,10):
    idx = np.random.randint(0, load_data.shape[0], 25)
    idx1 = np.random.randint(0, load_data.shape[0], 25)
    fid = calculate_fid(load_data[idx], load_data[idx1])
    real_fid.append(fid)
print(real_fid)
print(sum(real_fid)/len(real_fid))
'''

#Plotting FID scores
print(fidDict)
fig, ax = plt.subplots()
line = ax.plot(fidDict["epoch"], fidDict["fid"], 'o', color='b')
plt.xlabel("Epoch")
plt.ylabel("FID score")
plt.show()
fig.savefig("FID.png")

#Save model for future use to generate fake images
#Not tested yet... make sure right model is being saved..
#Compare with GAN4

generator.save('generator_model.h5')  #Test the model on GAN4_predict... currently, issues with saving the model
#Change epochs back to 30K
                


