print('--------Importing Modules and Packages--------')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf 
from layer.discriminator32 import Discriminator512
from layer.generator32 import Generator512
#from layer.plotter import generate_and_save_images
import numpy as np 
from layer.dataloader import *
import time

print('--------Successfully Imported--------')

start = time.process_time() 

NUM_EXP_TO_GENERATE = 1
CKPT_NUM = 1490
KERNEL_SIZE = (3, 3)
GENERATOR_INPUT = (4, 4, 16) # (4, 4, 16) --> 64, 64      (8, 8, 16) --> 128, 128      (16, 16, 16) --> 256, 256
LR_GEN = 1e-4  #1e-4
LR_DISC = 4e-4 #4e-4
SEED = tf.random.normal([NUM_EXP_TO_GENERATE, *GENERATOR_INPUT])
OUTPUT_PATH  = './outputs/'
CKPT_DIR = './ADAM/ckpts/'
BATCH_SIZE_GENERATION = 1
NUM_BATCHES = int(NUM_EXP_TO_GENERATE/BATCH_SIZE_GENERATION)

generator_optimizer = tf.keras.optimizers.Adam(LR_GEN)
discriminator_optimizer = tf.keras.optimizers.Adam(LR_DISC)

#generator_optimizer = tf.keras.optimizers.RMSprop(LR_GEN)
#discriminator_optimizer = tf.keras.optimizers.RMSprop(LR_DISC)

generator = Generator512(generator_input_shape=GENERATOR_INPUT, kernel_size=(7, 7))
discriminator = Discriminator512(discriminator_input_shape=(64, 64, 1), kernel_size=(3, 3))


checkpoint_prefix = os.path.join(CKPT_DIR, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
                                 
print('--------Loding Model--------')
checkpoint.restore(CKPT_DIR + '/ckpt-' + str(CKPT_NUM))
print('--------Model Loaded Successfully--------')

def f(x):
    return np.base_repr(x)

def shift_b(x):
    return x[-8:len(x)]

def shift_g(x):
    return x[-16:len(x) - 8]

def to_decimal(x):
    if x == '':
        x = '0'
    return int(x, 2)


def generate_and_save_images(model, test_input, file_path, batch_num, batch_size):
    print('--------Starting Predictions--------')
    predictions = model(test_input, training=False).numpy()
    print('--------Finished Predictions--------')
    #print(predictions)
    #px = 1 / plt.rcParams['figure.dpi']
    #fig = plt.figure(figsize=(256 * px *8, 256 * px*8))
    print('--------Saving Images--------')
    print('Output Path: ', file_path)
    for i in range(predictions.shape[0]):
        #plt.subplot(4, 4, i + 1)
        denorm = (predictions[i, :, :] * 32767.5 + 32767.5).astype('float64')
        
        array_raster_binary = np.vectorize(f)(np.array(np.around(denorm)).astype(int))

        b_channel = np.vectorize(shift_b)(array_raster_binary)
        b_channel = np.vectorize(to_decimal)(b_channel)

        # reconstruct green
        g_channel = np.vectorize(shift_g)(array_raster_binary)
        g_channel = np.vectorize(to_decimal)(g_channel)

        # reconstruct red -> is always 1
        r_channel = np.array(g_channel)
        r_channel.fill(1)

        image_raster = np.dstack((r_channel.astype('uint8'),
                                g_channel.astype('uint8'), 
                                b_channel.astype('uint8')))
                                
        write_image( file_path + str( batch_num * batch_size + i ) + '.png', image_raster)
        print('image: ', batch_num * batch_size + i)

for i in range(0, NUM_BATCHES):
    print('batch: ',i)
    seed_batch = SEED[i*BATCH_SIZE_GENERATION:(i+1)*BATCH_SIZE_GENERATION, :, :, :]
    generate_and_save_images(checkpoint.generator, seed_batch, OUTPUT_PATH, i, BATCH_SIZE_GENERATION)
print('--------All Images Saved--------')
print(time.process_time() - start)
