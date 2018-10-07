import cv2
import numpy as np

TRAIN_MEAN = 13.378813743591309
TRAIN_STD = 25.18636131286621


def get_channel_filename(image_id, channel):
    return '{image_id}_{channel}.png'.format(image_id=image_id, channel=channel)


def load_channel(path, image_size):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)


def merge_channels(channels):
    image = np.array(channels)
    image = np.swapaxes(image, 0, 2)
    return image


def augment_image(image_data_generator, image):
    return image_data_generator.random_transform(image)


def preprocess(image_or_images):
    image_or_images -= TRAIN_MEAN
    image_or_images /= TRAIN_STD
