import tensorflow.keras as keras


class Discriminator1024(keras.Sequential):

    def __init__(self, discriminator_input_shape=(64, 64, 3), kernel_size=(3, 3)):
        super(Discriminator1024, self).__init__(name='discriminator1024')
        # Block-1
        self.add(keras.layers.Conv2D(64, kernel_size, strides=(2, 2), 
                                    padding='same', input_shape=(None, None, discriminator_input_shape[2])))
        self.add(keras.layers.LeakyReLU(0.25))
        self.add(keras.layers.Dropout(0.3))
        # Block-2
        self.add(keras.layers.Conv2D(128, kernel_size, strides=(2, 2), padding='same'))
        self.add(keras.layers.LeakyReLU(0.25))
        self.add(keras.layers.Dropout(0.3))
        # Block-2
        self.add(keras.layers.Conv2D(256, kernel_size, strides=(2, 2), padding='same'))
        self.add(keras.layers.LeakyReLU(0.25))
        self.add(keras.layers.Dropout(0.3))
        # Block-3
        self.add(keras.layers.Conv2D(512, kernel_size, strides=(2, 2), padding='same'))
        self.add(keras.layers.LeakyReLU(0.25))
        self.add(keras.layers.Dropout(0.3))
        # Block-4
        self.add(keras.layers.Conv2D(1024, kernel_size, strides=(2, 2), padding='same'))
        self.add(keras.layers.LeakyReLU(0.25))
        self.add(keras.layers.Dropout(0.3))
        # Block-5 discriminator output
        self.add(keras.layers.Conv2D(1, kernel_size=(2, 2), strides=(1, 1), padding='valid'))
        self.add(keras.layers.Activation('sigmoid'))
