import math
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import cv2


def read_image(labels, paths, height, width):
    idx = 0
    list_to_save_images = list()
    list_to_save_labels = list()
    list_to_save_filenames = list()
    for filename in paths:
        im = cv2.imread(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # decuparea fetelor in functie de coordonate
        x1 = float(labels[labels['path'] == filename]['face_location_x1'].to_list()[0])
        x2 = float(labels[labels['path'] == filename]['face_location_x2'].to_list()[0])
        y1 = float(labels[labels['path'] == filename]['face_location_y1'].to_list()[0])
        y2 = float(labels[labels['path'] == filename]['face_location_y2'].to_list()[0])

        im = im[math.floor(x2):math.floor(y2), math.floor(x1):math.floor(y1)]
        im = cv2.resize(im, (height, width), interpolation=cv2.INTER_CUBIC)

        list_to_save_images.append(im)
        list_to_save_labels.append(float(labels[labels['path'] == filename]['gender'].to_list()[0]))
        list_to_save_filenames.append(filename)

        idx += 1
        if idx % 100 == 0:
            print('Processed {}/{}'.format(idx, len(paths)))

    return np.array(list_to_save_images), np.array(list_to_save_labels), np.array(list_to_save_filenames)


def get_images(height, width, total_images):
    # importare etichete din csv
    labels = pd.read_csv(r'C:\Users\Cera\PycharmProjects\pythonProject\procesare\processed_meta_v4.csv', dtype=str)
    labels = labels[:total_images]

    # afisarea numarului de barbati si femei, pregatirea celor 3 seturi de date
    # males = labels[labels['gender'] == '0']
    # females = labels[labels['gender'] == '1']
    # print('\nNumber of Males Images: ', len(males))
    # print('Number of Females Images: ', len(females))

    traindf = labels[:math.floor(0.6 * total_images)]
    validdf = labels[math.floor(0.6 * total_images):math.floor(0.8 * total_images)]
    testdf = labels[math.floor(0.8 * total_images):total_images]

    train_images, train_labels, train_paths = read_image(labels, traindf.path, height, width)
    valid_images, valid_labels, valid_paths = read_image(labels, validdf.path, height, width)
    test_images, test_labels, test_paths = read_image(labels, testdf.path, height, width)

    return train_images, to_categorical(train_labels), train_paths, \
           valid_images, to_categorical(valid_labels), valid_paths, \
           test_images, to_categorical(test_labels), test_paths
