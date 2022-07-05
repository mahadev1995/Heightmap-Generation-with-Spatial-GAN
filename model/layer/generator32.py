import tensorflow.keras as keras

class Generator512(keras.Sequential):

    def __init__(self, generator_input_shape=(4, 4 ,4), kernel_size=(3, 3)):
        super(Generator512, self).__init__(name="generator512")
        # Block-1
        self.add(keras.layers.Conv2D(512, kernel_size,
                                              padding='same', use_bias=True, input_shape=(None, None, generator_input_shape[2])))
        self.add(keras.layers.BatchNormalization())
        self.add(keras.layers.LeakyReLU())
        # Block-2
        self.add(keras.layers.UpSampling2D(size=(2,2), interpolation='nearest'))
        self.add(keras.layers.Conv2D(256, kernel_size, padding='same', use_bias=True))
        self.add(keras.layers.BatchNormalization())
        self.add(keras.layers.LeakyReLU(0.25))       
        # Block-3
        self.add(keras.layers.UpSampling2D(size=(2,2), interpolation='nearest'))
        self.add(keras.layers.Conv2D(128, kernel_size, padding='same', use_bias=True))
        self.add(keras.layers.BatchNormalization())
        self.add(keras.layers.LeakyReLU(0.25))
        # Block-4
        self.add(keras.layers.UpSampling2D(size=(2,2), interpolation='nearest'))
        self.add(keras.layers.Conv2D(64, kernel_size, padding='same', use_bias=True))
        self.add(keras.layers.BatchNormalization())
        self.add(keras.layers.LeakyReLU(0.25))
        # Block-5 
        self.add(keras.layers.UpSampling2D(size=(2,2), interpolation='nearest'))
        self.add(keras.layers.Conv2D(32, kernel_size, padding='same', use_bias=True))
        self.add(keras.layers.BatchNormalization())
        self.add(keras.layers.LeakyReLU(0.25))
        # Block6 -- Generator Output
        self.add(keras.layers.Conv2D(1, kernel_size, padding='same', use_bias=True, activation='tanh'))
