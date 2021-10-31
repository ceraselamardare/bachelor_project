import math
from keras.utils import to_categorical

import pandas as pd
import numpy as np
import cv2


def read_image(labels, paths, height, width):
    path = r'C:/Users/Cera/PycharmProjects/pythonProject/celeb_faces/img_align_celeba/img_align_celeba/'
    idx = 0
    list_to_save_images = list()
    list_to_save_labels = list()
    list_to_save_filenames = list()
    list_to_save_labels_hair_colour = list()
    for filename in paths:
        im = cv2.imread(path + filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # plt.imshow(im)
        # plt.show()

        # decuparea fetelor in functie de coordonate
        im = cv2.resize(im, (height, width), interpolation=cv2.INTER_CUBIC)

        list_to_save_images.append(im)
        list_to_save_filenames.append(filename)

        if int(labels[labels['image_id'] == filename]['bald'].to_list()[0]):
            list_to_save_labels_hair_colour.append(0)
        elif int(labels[labels['image_id'] == filename]['black_hair'].to_list()[0]):
            list_to_save_labels_hair_colour.append(1)
        elif int(labels[labels['image_id'] == filename]['blond_hair'].to_list()[0]):
            list_to_save_labels_hair_colour.append(2)
        elif int(labels[labels['image_id'] == filename]['brown_hair'].to_list()[0]):
            list_to_save_labels_hair_colour.append(3)
        elif int(labels[labels['image_id'] == filename]['gray_hair'].to_list()[0]):
            list_to_save_labels_hair_colour.append(4)
        else:
            print('EROROROROROROR')

        idx += 1
        if idx % 100 == 0:
            print('Processed {}/{}'.format(idx, len(paths)))

    return np.array(list_to_save_images), np.array(list_to_save_labels_hair_colour), np.array(list_to_save_filenames)


def get_images(height, width, total_images):
    # importare etichete din csv
    labels = pd.read_csv(r'C:\Users\Cera\PycharmProjects\pythonProject\procesare\processed_celeb_faces.csv', dtype=str)
    labels = labels[:total_images]

    traindf = labels[:math.floor(0.6 * total_images)]
    validdf = labels[math.floor(0.6 * total_images):math.floor(0.8 * total_images)]
    testdf = labels[math.floor(0.8 * total_images):total_images]

    train_images, train_labels, train_paths = read_image(labels, traindf.image_id, height, width)
    valid_images, valid_labels, valid_paths = read_image(labels, validdf.image_id, height, width)
    test_images, test_labels, test_paths = read_image(labels, testdf.image_id, height, width)

    return train_images, to_categorical(train_labels), train_paths, \
           valid_images, to_categorical(valid_labels), valid_paths, \
           test_images, to_categorical(test_labels), test_paths
