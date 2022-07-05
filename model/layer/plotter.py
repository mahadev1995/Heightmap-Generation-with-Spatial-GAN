import numpy as np 
import matplotlib.pyplot as plt


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

def generate_and_save_images(model, test_input, file_path):
    predictions = model(test_input, training=False).numpy()
    #print(predictions)
    #px = 1 / plt.rcParams['figure.dpi']
    #fig = plt.figure(figsize=(256 * px *8, 256 * px*8))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
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
        plt.imshow(image_raster)
        
        plt.axis('off')

    #plt.savefig(file_path)
    #plt.show(block=False)
    #plt.pause(1)
    plt.savefig(file_path)
    plt.close()

