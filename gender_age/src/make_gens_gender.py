from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from gender_age.src.helpers.get_bs import get_bs
from gender_age.src.helpers.get_images_gender import get_images


def make_gens(height, width, batch_size, total_images):
    train_images, train_labels, train_paths, \
    valid_images, valid_labels, valid_paths, \
    test_images, test_labels, test_paths = get_images(height, width, total_images)

    train_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
                                   horizontal_flip=True).flow(x=train_images,
                                                              y=train_labels,
                                                              batch_size=batch_size, seed=123,
                                                              shuffle=True)
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

    return train_gen, test_gen, valid_gen
