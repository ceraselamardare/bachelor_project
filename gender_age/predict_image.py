import os
import numpy as np
import cv2
import tensorflow as tf
import numpy

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from gender_age.create_new_image_angle import plot_landmarks_angle

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def prepare(filepath):
    im = cv2.imread(filepath)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    IMG_SIZE = 224
    return cv2.resize(im, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)


def prepare_angle(filepath):
    im = cv2.imread(filepath)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (178, 218), interpolation=cv2.INTER_CUBIC)


def prepare_gender_age(filepath):
    face_cascade = cv2.CascadeClassifier(
        r'C:\Users\Cera\PycharmProjects\pythonProject\haarcascade_frontalface_default.xml')
    im = cv2.imread(filepath)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        print('No faces')
        return prepare(filepath)
    elif len(faces) > 1:
        print('More faces')
        return prepare(filepath)
        # faces = [faces[0]]
    for (x, y, w, h) in faces:
        # cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        im = im[y:y + h, x:x + w]
        # cv2.imshow('im', cv2.resize(im, (720, 1080), interpolation=cv2.INTER_CUBIC))
        # cv2.waitKey()
        # plt.show()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    IMG_SIZE = 224
    # im = cv2.resize(im, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    # plt.imshow(im)
    # plt.show()

    return cv2.resize(im, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)


def predict_gender(filepath):
    model = tf.keras.models.load_model(
        r'C:\Users\Cera\PycharmProjects\pythonProject\gender_age\save_models_gender\Mobilenet-84.57.h5')

    to_predict = np.array([prepare_gender_age(filepath)])
    test_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
                                  horizontal_flip=False).flow(x=to_predict,
                                                              batch_size=1,
                                                              shuffle=False)

    prediction = model.predict(test_gen, steps=None)
    new_dict = {
        0: 'male',
        1: 'female'
    }

    return new_dict[numpy.argmax(prediction[0])]


def predict_age(filepath):
    model = tf.keras.models.load_model(
        r'C:\Users\Cera\PycharmProjects\pythonProject\gender_age\save_models\best_Mobilenet-67.71.h5')

    to_predict = np.array([prepare_gender_age(filepath)])
    test_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
                                  horizontal_flip=False).flow(x=to_predict,
                                                              batch_size=1,
                                                              shuffle=False)
    prediction = model.predict(test_gen, steps=None)

    new_dict = {
        0: '0-18',
        1: '18-35',
        2: '35-65',
        3: '65+',
    }

    return new_dict[numpy.argmax(prediction[0])]


def predict_hair(filepath):
    model = tf.keras.models.load_model(
        r'C:\Users\Cera\PycharmProjects\pythonProject\gender_age\save_models_hair\MobilenetV2-89.80.h5')

    to_predict = np.array([prepare(filepath)])
    test_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
                                  horizontal_flip=False).flow(x=to_predict,
                                                              batch_size=1,
                                                              shuffle=False)
    prediction = model.predict(test_gen, steps=None)

    new_dict = {
        0: 'bald hair',
        1: 'black hair',
        2: 'blond hair',
        3: 'brown hair',
        4: 'gray hair',
    }

    return new_dict[numpy.argmax(prediction[0])]


def predict_angle(filepath):
    model = tf.keras.models.load_model(
        r'C:\Users\Cera\PycharmProjects\pythonProject\gender_age\save_models_angle\Mobilenet-1.64-best.h5')

    to_predict = np.array([prepare_angle(filepath)])
    test_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
                                  horizontal_flip=False).flow(x=to_predict,
                                                              batch_size=1,
                                                              shuffle=False)
    prediction = model.predict(test_gen, steps=None)
    plot_landmarks_angle(filepath, prediction)


def predict_landmarks_cv2(filepath):
    model = tf.keras.models.load_model(
        r'C:\Users\Cera\PycharmProjects\pythonProject\gender_age\save_models_angle\VGG19-3.29-cv2.h5')

    face_cascade = cv2.CascadeClassifier(
        r'C:\Users\Cera\PycharmProjects\pythonProject\haarcascade_frontalface_default.xml')

    im = cv2.imread(filepath)

    height_old, width_old, _ = im.shape

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print('No faces')

    elif len(faces) > 1:
        print('More faces')

    prediction = []
    for (x, y, w, h) in faces:
        # plt.imshow(im)
        # plt.show()

        img = im[y:y + h, x:x + w]
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

        # plt.imshow(img)
        # plt.show()

        prediction = model.predict(np.array([img]), steps=None)
        print(prediction)

        prediction = prediction[0]

        #############################################Print points
        image_points = [
            (prediction[0], prediction[1]),
            (prediction[2], prediction[3]),
            (prediction[4], prediction[5]),
            (prediction[6], prediction[7]),
            (prediction[8], prediction[9]),
        ]

        # for p in image_points:
        #     cv2.circle(img, (int(p[0]), int(p[1])), 3, (154, 132, 0), -1)

        # cv2.imshow('img', img)
        # cv2.waitKey()
        #########################################

        shape_x = w
        shape_y = h

        # NewX = X * NewWidth / OldWidth
        for idx in range(len(prediction)):
            if idx % 2 == 0:  # punct pe x
                prediction[idx] = prediction[idx] * shape_x / img.shape[0]
                prediction[idx] += x
            else:  # punct pe y
                prediction[idx] = prediction[idx] * shape_y / img.shape[1]
                prediction[idx] += y

        break

    shape_x = width_old
    shape_y = height_old

    # NewX = X * NewWidth / OldWidth
    for idx in range(len(prediction)):
        if idx % 2 == 0:  # punct pe x
            prediction[idx] = prediction[idx] * shape_x / im.shape[1]
        else:  # punct pe y
            prediction[idx] = prediction[idx] * shape_y / im.shape[0]

    # im = cv2.resize(im, (shape_x, shape_y), interpolation=cv2.INTER_CUBIC)
    #
    #
    # image_points = [
    #     (prediction[0], prediction[1]),
    #     (prediction[2], prediction[3]),
    #     (prediction[4], prediction[5]),
    #     (prediction[6], prediction[7]),
    #     (prediction[8], prediction[9]),
    # ]
    #
    # for p in image_points:
    #     cv2.circle(im, (int(p[0]), int(p[1])), 3, (154, 132, 0), -1)

    # cv2.imshow('img', im)
    # cv2.waitKey()

    return prediction


def predict_angle_v2(filepath):
    prediction = predict_landmarks_cv2(filepath)
    angle = plot_landmarks_angle(filepath, prediction)

    return angle
