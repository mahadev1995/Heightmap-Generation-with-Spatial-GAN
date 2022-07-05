print('++++++++++++++ Loading Modules and Packages +++++++++++++++++')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf 
from layer.discriminator32 import Discriminator512
from layer.generator32 import Generator512
from layer.plotter import generate_and_save_images
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow_addons.optimizers import AdamW
print('++++++++++++++ Successfully Loaded +++++++++++++++++')
NUM_EXP_TO_GENERATE = 16

# KERNEL_SIZE = (5, 5)
GENERATOR_INPUT = (4, 4, 16)
LR_GEN = 1e-4  #1e-4
LR_DISC = 4e-4 #4e-4
START_EPOCH = 1490
END_EPOCH = 1490
OUT_PATH = './ADAM/output/output3_3_'
SEED = tf.random.normal([NUM_EXP_TO_GENERATE, *GENERATOR_INPUT])

# generator_optimizer = tf.keras.optimizers.RMSprop(LEARNING_RATE)
# discriminator_optimizer = tf.keras.optimizers.RMSprop(LEARNING_RATE)
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#                                                             initial_learning_rate=1e-5,
#                                                             decay_steps=5000,
#                                                             decay_rate=0.9
#                                                           )
# lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([8000], [1e-5, 5e-6])
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_GEN, beta_1=0.9, beta_2=0.99)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_DISC, beta_1=0.9, beta_2=0.99)
# generator_optimizer = AdamW(weight_decay=1e-5, learning_rate=LR_GEN, beta_1=0.9, beta_2=0.99, amsgrad=False)
# discriminator_optimizer = AdamW(weight_decay=1e-5, learning_rate=LR_DISC, beta_1=0.9, beta_2=0.99, amsgrad=False)
# generator_optimizer = tf.keras.optimizers.RMSprop(LR_GEN)
# discriminator_optimizer = tf.keras.optimizers.RMSprop(LR_DISC)
# generator_optimizer = AdamW(weight_decay=1e-5, learning_rate=LR_GEN, beta_1=0.9, beta_2=0.99, amsgrad=False)
# discriminator_optimizer = AdamW(weight_decay=1e-5, learning_rate=LR_DISC, beta_1=0.9, beta_2=0.99, amsgrad=False)

generator = Generator512(generator_input_shape=GENERATOR_INPUT, kernel_size=(7, 7))
discriminator = Discriminator512(discriminator_input_shape=(64, 64, 1), kernel_size=(3, 3))

checkpoint_dir = './ADAM/ckpts/' 
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


for i in range(START_EPOCH, END_EPOCH+1):

    checkpoint.restore(checkpoint_dir + '/ckpt-' + str(i))
    output_path = OUT_PATH + str(i)

    generate_and_save_images(checkpoint.generator,
                            SEED,
                            output_path)
    print('Checkpoint: ', i)
