import tensorflow as tf
from layer.loss import *
from tqdm import tqdm

@tf.function
def train_step(images, generator, discriminator, cross_entropy, generator_optimizer, discriminator_optimizer, batch_size):
    noise = tf.random.normal([batch_size, 8, 8, 1])
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


def train(dataset, generator, discriminator, cross_entropy, generator_optimizer,  discriminator_optimizer, batch_size, epochs, steps_per_epoch = 2000):
    outer = tqdm(total=epochs, desc='EPOCH', leave=False, position=0)
    for epoch in range(epochs):
        #start = time.time()
        #print("\nepoch {}/{}".format(epoch+1,epochs))
        inner = tqdm(total=steps_per_epoch, desc='Steps', leave=False, position=1)
        step = 0
        #with tqdm(dataset, unit='batch') as tepoch:
        for image in dataset:
            #print(image.shape)
            train_step(image, generator, discriminator, cross_entropy,generator_optimizer, discriminator_optimizer, batch_size)
            #print(step)
            inner.update(1)
            if step == steps_per_epoch:
                inner.close()
                break
            step = step + 1

        outer.update(1)
        checkpoint.save(file_prefix=checkpoint_prefix)
        #if (epoch)%40 == 0:
        #generate_and_save_images(generator, epoch+1, seed)
            #checkpoint.save(file_prefix=checkpoint_prefix) 
    #display.clear_output(wait=True)
    # generate_and_save_images(generator,
    #                          epochs,
    #                          seed)
     
