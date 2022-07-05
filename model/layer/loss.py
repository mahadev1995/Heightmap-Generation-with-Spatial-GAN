import tensorflow as tf


def discriminator_loss(real_output, fake_output, entropy):
    real_loss = entropy(0.9*tf.ones_like(real_output),real_output)
    fake_loss = entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output, entropy):
    return entropy(tf.ones_like(fake_output), fake_output)
