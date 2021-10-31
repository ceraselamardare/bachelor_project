import math
from keras.utils import to_categorical

import pandas as pd
import numpy as np
import cv2



def crop_and_modify_landmarks(img, landmarks, shape, path):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        r'C:\Users\Cera\PycharmProjects\pythonProject\haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) < 1:
        return None, None

    sw = True

    for (x, y, w, h) in faces:
        landmarks_aux = landmarks.copy()
        im = img[y:y + h, x:x + w]

        sw = True

        for idx in range(len(landmarks_aux)):
            if idx % 2 == 0:  # punct pe x
                if not x <= landmarks[idx] <= x + w:
                    break
                landmarks_aux[idx] -= x
                landmarks_aux[idx] = landmarks_aux[idx] * shape / im.shape[0]
            else:  # punct pe y
                if not y <= landmarks_aux[idx] <= y + h:
                    sw = False
                    break
                landmarks_aux[idx] -= y
                landmarks_aux[idx] = landmarks_aux[idx] * shape / im.shape[1]

        if sw:
            img = img[y:y + h, x:x + w]
            img = cv2.resize(img, (shape, shape), interpolation=cv2.INTER_CUBIC)
            landmarks = landmarks_aux
            break

    if not sw:
        return None, None

    return img, landmarks


def read_image(labels, paths, height, width):
    path = r'C:/Users/Cera/PycharmProjects/pythonProject/celeb_faces/img_align_celeba/img_align_celeba/'
    idx = 0
    list_to_save_images = list()
    list_to_save_labels = list()
    list_to_save_filenames = list()
    list_to_save_labels_angle = list()

    for filename in paths:
        im = cv2.imread(path + filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # plt.imshow(im)
        # plt.show()

        # decuparea fetelor in functie de coordonate
        # im = cv2.resize(im, (width, height), interpolation=cv2.INTER_CUBIC)

        label = list()
        label = [
            float(labels[labels['image_id'] == filename]['lefteye_x'].to_list()[0]),
            float(labels[labels['image_id'] == filename]['lefteye_y'].to_list()[0]),
            float(labels[labels['image_id'] == filename]['righteye_x'].to_list()[0]),
            float(labels[labels['image_id'] == filename]['righteye_y'].to_list()[0]),
            float(labels[labels['image_id'] == filename]['nose_x'].to_list()[0]),
            float(labels[labels['image_id'] == filename]['nose_y'].to_list()[0]),
            float(labels[labels['image_id'] == filename]['leftmouth_x'].to_list()[0]),
            float(labels[labels['image_id'] == filename]['leftmouth_y'].to_list()[0]),
            float(labels[labels['image_id'] == filename]['rightmouth_x'].to_list()[0]),
            float(labels[labels['image_id'] == filename]['rightmouth_y'].to_list()[0]),
        ]

        im, label = crop_and_modify_landmarks(im, label, 224, filename)
        if im is None:
            continue

        list_to_save_images.append(im)
        list_to_save_labels_angle.append(label)
        list_to_save_filenames.append(filename)

        idx += 1
        if idx % 100 == 0:
            print('Processed {}/{}'.format(idx, len(paths)))

    print('Total images with landmarks: {}'.format(len(list_to_save_images)))

    return np.array(list_to_save_images), np.array(list_to_save_labels_angle), np.array(list_to_save_filenames)


def get_images(height, width, total_images):
    # importare etichete din csv
    labels = pd.read_csv(r'C:\Users\Cera\PycharmProjects\pythonProject\procesare\list_landmarks_align_celeba.csv',
                         dtype=str)
    labels = labels[:total_images]

    traindf = labels[:math.floor(0.6 * total_images)]
    validdf = labels[math.floor(0.6 * total_images):math.floor(0.8 * total_images)]
    testdf = labels[math.floor(0.8 * total_images):total_images]

    train_images, train_labels, train_paths = read_image(labels, traindf.image_id, height, width)
    valid_images, valid_labels, valid_paths = read_image(labels, validdf.image_id, height, width)
    test_images, test_labels, test_paths = read_image(labels, testdf.image_id, height, width)

    return train_images, train_labels, train_paths, \
           valid_images, valid_labels, valid_paths, \
           test_images, test_labels, test_paths
