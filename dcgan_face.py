from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import matplotlib.image as matimg
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import glob
import os
import numpy as np
from PIL import Image
import argparse
import math
from keras.optimizers import Adam

def load_image(path):
    img = matimg.imread(path)
    return img

#following code loads data from local repo
#directory should have two subfolders, with name train and test
def load_local_data(path):
    paths = glob.glob(os.path.join(path + "/train", "*.jpg"))
    X_train = np.array( [ load_image(p) for p in paths ] )

    paths = glob.glob(os.path.join(path + "/test", "*.jpg"))
    X_test = np.array( [ load_image(p) for p in paths ] )
   
    return X_train, X_test

#Note: This code is for 32x32 colored image
def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*8*8))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((128, 8, 8), input_shape=(128*8*8,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    return model

#Note: This code is for 32x32 colored image
def discriminator_model():
    model = Sequential()
    model.add(Convolution2D(
                        64, 5, 5,
                        border_mode='same',
                        input_shape=(3, 32, 32)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 5, 5))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

    #dummy_input = np.ones((1, 1, 28, 28))   
    # For TensorFlow dummy_input = np.ones((32, 12, 12, 3))
    #preds = model.predict(dummy_input)
    #print(preds.shape)

def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    generated_images = generated_images.reshape((generated_images.shape[0], 32, 32, 3) )
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1], 3),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1], 0] = \
            img[:, :, 0]
	image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1], 1] = \
            img[:, :, 1]
	image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1], 2] = \
            img[:, :, 2]
    return image


def train(BATCH_SIZE):
    #path to dataset
    path="Project/data/celebD"
    (X_train, X_test) = load_local_data(path)

    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    print X_train.shape
    X_train = X_train.reshape((X_train.shape[0], 3) + X_train.shape[1:3])
    print X_train.shape
    discriminator = discriminator_model()
    generator = generator_model()
    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)
    d_optim = Adam(lr=0.0002, beta_1=0.5)
    g_optim = Adam(lr=0.0002, beta_1=0.5)
    generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator_on_generator.compile(
        loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, 100))

    for epoch in range(10):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
	# load weights on first try (i.e. if process failed previously and we are attempting to recapture lost data)
        if epoch == 0:
            if os.path.exists('generator') and os.path.exists('discriminator'):
                print "Loading saves weights.."
                generator.load_weights('generator')
                discriminator.load_weights('discriminator')
                print "Finished loading"
            else:
                pass

        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = generator.predict(noise, verbose=0)
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch)+"_"+str(index)+".png")
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * BATCH_SIZE)
            discriminator.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)

def generate(BATCH_SIZE):
    generator = generator_model()
    d_optim = Adam(lr=0.0002, beta_1=0.5)
    g_optim = Adam(lr=0.0002, beta_1=0.5)
    generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    generator.load_weights('generator')
    noise = np.zeros((BATCH_SIZE, 100))
    for i in range(BATCH_SIZE):
    	noise[i, :] = np.random.uniform(-1, 1, 100)
    generated_images = generator.predict(noise, verbose=1)
    image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")

#to run this code give this command: "python dcgan_face.py --mode train --batch_size 128"
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size)
