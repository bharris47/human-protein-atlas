import csv
from functools import partial
from multiprocessing.pool import ThreadPool
from os.path import join

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from protein_atlas.image import load_channel, get_channel_filename


def load_data(instances_path, image_directory, image_size=224):
    with open(instances_path) as f:
        reader = csv.reader(f)
        next(reader)  # ignore header
        instances = list(reader)

    image_ids, label_strings = zip(*instances)
    image_paths = [join(image_directory, get_channel_filename(image_id, 'green')) for image_id in image_ids]
    images = np.zeros(shape=(len(instances), image_size, image_size, 1), dtype=np.float32)
    with ThreadPool() as pool:
        load_func = partial(load_channel, image_size=image_size)
        for i, image in tqdm(enumerate(pool.imap(load_func, image_paths)), total=len(image_paths)):
            images[i, :, :, 0] = image

    labels = [tuple(int(label) for label in label_string.split()) for label_string in label_strings]
    labels = MultiLabelBinarizer().fit_transform(labels)
    return images, labels
