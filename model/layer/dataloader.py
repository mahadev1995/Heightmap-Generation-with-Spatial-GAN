import glob
import tensorflow as tf
import tensorflow.keras as keras

def load_image(file_name: str):
    img_string = tf.io.read_file(file_name)
    return tf.image.decode_image(img_string, channels=3)

def write_image(file_name: str, img):
    image_string = tf.image.encode_png(img)
    tf.io.write_file(file_name, image_string)

def normalize(image):
  return tf.expand_dims((image[:, :, 1]*256 - image[:, :, 2] - 32767.5) / 32767.5, axis = -1)

def crop_image(image, patch_size: tuple):
    return tf.cast( tf.image.random_crop(image, (patch_size, patch_size, 3)), tf.float32 )

def generate_dataset(mode: str, file_path: str, batch_size: int, 
                                                patch_size: tuple, num_threads: int):
    with tf.device("/cpu:0"):
        images = glob.glob(file_path)
        print("Number of images found: " + str(len(images)))
        if not images:
            raise RuntimeError(f"No training images found in the directory "
                               f"'{file_path}'.")

        data = tf.data.Dataset.from_tensor_slices(images).shuffle(len(images), reshuffle_each_iteration=True)

        if mode.upper() == "TRAIN":
            data = data.repeat()

        data = data.map(
            lambda x: normalize(crop_image(load_image(x), patch_size)),
            num_parallel_calls=num_threads
        )

        data = data.batch(batch_size, drop_remainder=True)

    return data
