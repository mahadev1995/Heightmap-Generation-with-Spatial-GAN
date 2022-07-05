import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import time
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import Progbar
#import matplotlib.pyplot as plt
from layer.discriminator32 import Discriminator256
from layer.generator32 import Generator256
from layer.loss import *
from layer.dataloader import *
#from layer.plotter import generate_and_save_images
from tqdm import tqdm
from tensorflow_addons.optimizers import AdamW
import pandas as pd
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

BATCH_SIZE = 64
STEPS_PER_EPOCH = 2185 #2000
         #60
KERNEL_SIZE = (3, 3)
GENERATOR_INPUT = (4, 4, 16)
LEARNING_RATE = 1e-7
START_EPOCH = 821
END_EPOCH = 2000
EPOCHS = int(END_EPOCH - START_EPOCH) 
LR_GEN = 1e-4 #1e-4
LR_DISC = 4e-4 #4e-4
#NOISE_CHANNELS = 8vim 
#num_examples_to_generate = 4
#seed = tf.random.normal([num_examples_to_generate, 8, 8, 1])
path = '../data/training_dataset/*.png'  #*.png'

# cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False)
# generator_optimizer = AdamW(weight_decay=1e-5, learning_rate=LR_GEN, beta_1=0.9, beta_2=0.99)
# discriminator_optimizer = AdamW(weight_decay=1e-5, learning_rate=LR_DISC, beta_1=0.9, beta_2=0.99)
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False)
generator_optimizer = keras.optimizers.Adam(learning_rate=LR_GEN, beta_1=0.9, beta_2=0.99)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=LR_DISC, beta_1=0.9, beta_2=0.99)

generator = Generator256(generator_input_shape=GENERATOR_INPUT, kernel_size=(7, 7))
discriminator = Discriminator256(discriminator_input_shape=(64, 64, 1), kernel_size=(3, 3))

print(generator.summary())
print(discriminator.summary())
print('--- Loading Data ---')

dataset = create_cusotm_dataset(state='train', file_path=path,
                                      batch_size=BATCH_SIZE,
                                      patch_size=64,
                                      threads=16,)
time.sleep(0.5)
print('========================================')
time.sleep(0.5)
print('--- Successfully Loaded ---')
time.sleep(0.5)
print('========================================')
time.sleep(0.5)
#checkpoint_dir = './training_checkpoints'
checkpoint_dir = './ADAM/ckpt'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint.restore(checkpoint_dir + '/ckpt-' + str(START_EPOCH))
ckpt_dir = './ADAM/ckpt'
ckpt_prfix = os.path.join(ckpt_dir, "ckpt")
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, *GENERATOR_INPUT])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = checkpoint.generator(noise, training=True)

        real_output = checkpoint.discriminator(images, training=True)
        fake_output = checkpoint.discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output, cross_entropy)
        disc_loss = discriminator_loss(real_output, fake_output, cross_entropy)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    checkpoint.generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    checkpoint.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return {'gen_loss': gen_loss, 
            'disc_loss': disc_loss, 
            'gen_lr': checkpoint.generator_optimizer._decayed_lr(tf.float32), 
            'disc_lr': checkpoint.discriminator_optimizer._decayed_lr(tf.float32)}

def train(dataset, epochs, start_epoch, end_epoch, steps_per_epoch = 2000):
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
        history.to_excel('./ADAM/history/history32.xlsx')

print('--- Started Training ---')
train(dataset,EPOCHS, START_EPOCH, END_EPOCH+1, STEPS_PER_EPOCH)
print('---Training Completed Successfully---')
