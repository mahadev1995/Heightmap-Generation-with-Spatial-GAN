import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import time
import tensorflow as tf
import tensorflow.keras as keras
from layer.discriminator32 import Discriminator512
from layer.generator32 import Generator512
from layer.loss import *
from layer.dataloader import *
from tqdm import tqdm
import numpy as np 
import pandas as pd 
from tensorflow_addons.optimizers import AdamW

np.random.seed(1234)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

BATCH_SIZE = 64             #32, 64, 128  
STEPS_PER_EPOCH = 2185      #2000 1171
EPOCHS = 2000               #60
# KERNEL_SIZE = (5, 5)
GENERATOR_INPUT = (4, 4, 16) #(4, 4, 16)
LR_GEN = 1e-4  #1e-4
LR_DISC = 4e-4 #4e-4

#num_examples_to_generate = 4
#seed = tf.random.normal([num_examples_to_generate, 8, 8, 1])
path = './data/training_dataset/*.png'  #*.png'
history_path = './ADAM/history.xlsx' #'./ADAM/history/history.xlsx'
#====================================================
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#                                                             initial_learning_rate=1e-5,
#                                                             decay_steps=5000,
#                                                             decay_rate=0.9
#                                                           )
# lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay([8000], [1e-5, 5e-6])
# TODO: Try AdamW done
# TODO: Try turning on amsgrad option to TRUE -- did not work
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False)
generator_optimizer = keras.optimizers.Adam(learning_rate=LR_GEN, beta_1=0.9, beta_2=0.99)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=LR_DISC, beta_1=0.9, beta_2=0.99)
# generator_optimizer = tf.keras.optimizers.RMSprop(LR_GEN)
# discriminator_optimizer = tf.keras.optimizers.RMSprop(LR_DISC)
# generator_optimizer = AdamW(weight_decay=1e-5, learning_rate=LR_GEN, beta_1=0.9, beta_2=0.99, amsgrad=False)
# discriminator_optimizer = AdamW(weight_decay=1e-5, learning_rate=LR_DISC, beta_1=0.9, beta_2=0.99, amsgrad=False)


generator = Generator512(generator_input_shape=GENERATOR_INPUT, kernel_size=(7, 7))
discriminator = Discriminator512(discriminator_input_shape=(64, 64, 1), kernel_size=(3, 3))

print(generator.summary())
print(discriminator.summary())
print('--- Loading Data ---')

dataset = generate_dataset(mode='train', file_path=path,
                                      batch_size=BATCH_SIZE,
                                      patch_size=64,
                                      num_threads=16,) #16
time.sleep(0.5)
print('========================================')
time.sleep(0.5)
print('--- Successfully Loaded ---')
time.sleep(0.5)
print('========================================')
time.sleep(0.5)
#checkpoint_dir = './training_checkpoints'
checkpoint_dir = './ADAM/ckpt_32'       #'./LR5_5/64/checkpoints16' for adamw use history_cleaned
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, *GENERATOR_INPUT])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output, cross_entropy)
        disc_loss = discriminator_loss(real_output, fake_output, cross_entropy)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return {'gen_loss': gen_loss, 
            'disc_loss': disc_loss, 
            'gen_lr': generator_optimizer._decayed_lr(tf.float32), 
            'disc_lr': discriminator_optimizer._decayed_lr(tf.float32)}

def train(dataset, epochs, steps_per_epoch = 2000):
    outer = tqdm(total=epochs, desc='EPOCH', leave=False, position=0)
    history = pd.DataFrame()
    for epoch in range(epochs):
        inner = tqdm(total=steps_per_epoch, desc='Steps', leave=False, position=1)
        step = 0
        for image in dataset:
            lr = train_step(image)
            
            inner.update(1)
            if step == steps_per_epoch:
                inner.close()
                break
            step = step + 1

        lr['gen_loss'] = lr['gen_loss'].numpy()
        lr['disc_loss'] = lr['disc_loss'].numpy()
        lr['gen_lr'] = lr['gen_lr'].numpy()
        lr['disc_lr'] = lr['disc_lr'].numpy()
        history=history.append(lr, ignore_index=True)
        # print(history)
        outer.update(1)
        checkpoint.save(file_prefix=checkpoint_prefix)
        history.to_excel(history_path)

    

print('--- Started Training ---')
train(dataset, EPOCHS, STEPS_PER_EPOCH)
print('---Training Completed Successfully---')
