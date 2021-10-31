from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import cv2
from gender_age.src.helpers.get_bs import get_bs
from gender_age.src.helpers.get_images_angle import get_images


def check_images(images_list, landmarks_list):
    for i in range(5, 10):
        img = images_list[i]
        point = landmarks_list[i]

        image_points = [
            (point[0], point[1]),
            (point[2], point[3]),
            (point[4], point[5]),
            (point[6], point[7]),
            (point[8], point[9]),
        ]

        for p in image_points:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (154, 132, 0), -1)

        cv2.imshow('img', img)
        cv2.waitKey()


def make_gens(height, width, batch_size, total_images):
    train_images, train_labels, train_paths, \
    valid_images, valid_labels, valid_paths, \
    test_images, test_labels, test_paths = get_images(height, width, total_images)

    # train_images, train_labels = angle_data_aug(train_images, train_labels)

    # check_images(train_images, train_labels)
    # check_images(valid_images, valid_labels)
    # check_images(test_images, test_labels)

    train_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
                                   horizontal_flip=False).flow(x=train_images,
                                                               y=train_labels,
                                                               batch_size=batch_size, seed=123,
                                                               shuffle=False)

    train_gen.filenames_paths_ex = train_paths

    valid_batch_size, valid_steps = get_bs(valid_labels, batch_size)

    valid_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
                                   horizontal_flip=False).flow(x=valid_images,
                                                               y=valid_labels,
                                                               batch_size=valid_batch_size,
                                                               shuffle=False)
    valid_gen.filenames_paths_ex = valid_paths

    test_batch_size, test_steps = get_bs(test_labels, batch_size)

    test_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
                                  horizontal_flip=False).flow(x=test_images,
                                                              y=test_labels,
                                                              batch_size=test_batch_size,
                                                              shuffle=False)
    test_gen.filenames_paths_ex = test_paths

    return train_images, train_labels, train_paths, \
           valid_images, valid_labels, valid_paths, \
           test_images, test_labels, test_paths
