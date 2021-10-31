from gender_age.src.helpers.print_in_color import print_in_color
import numpy as np

import cv2
from matplotlib import pyplot as plt
import numpy as np

import matplotlib.pyplot as plt


def print_info(labels,file_names, preds, print_code, max_err=0.03):
    error_list = []
    true_class = []
    pred_class = []

    total_correct = 0
    total_with_error = 0

    samples_correct = []
    samples_with_error = []

    new_dict = {
        0: 'lefteye_x',
        1: 'lefteye_y',
        2: 'righteye_x',
        3: 'righteye_y',
        4: 'nose_x',
        5: 'nose_y',
        6: 'leftmouth_x',
        7: 'leftmouth_y',
        8: 'rightmouth_x',
        9: 'rightmouth_y',
    }
    error_indices = []
    errors = 0
    for i, p in enumerate(preds):
        bad_labels = []
        for idx in range(len(labels[i])):
            val_true = labels[i][idx]
            val_pred = p[idx]
            err = abs(val_true-val_pred)/val_true
            if err > max_err:
                bad_labels.append(idx)

        if bad_labels:
            total_with_error += 1
            if len(samples_with_error) < 10:
                samples_with_error.append(
                    {
                        'photo': file_names[i],
                        'val_true': [round(val) for val in labels[i]],
                        'val_pred': [round(val) for val in p]
                    }
                )
        else:
            total_correct += 1
            if len(samples_correct) < 10:
                samples_correct.append(
                    {
                        'photo': file_names[i],
                        'val_true': [round(val) for val in labels[i]],
                        'val_pred': [round(val) for val in p]
                    }
                )

        if bad_labels:  # a misclassification has occurred
            error_list.append(file_names[i].replace('C:/Users/Cera/PycharmProjects/pythonProject/celeb_faces/img_align_celeba/img_align_celeba/', ''))
            true_classes = []
            pred_classes = []
            for idx in bad_labels:
                true_classes.append((new_dict[idx], labels[i][idx]))
                pred_classes.append((new_dict[idx], p[idx]))
            true_class.append(str(true_classes))
            pred_class.append(str(pred_classes))
            error_indices.append(val_pred)
            errors = errors + 1
    if print_code != 0:
        if errors > 0:
            if print_code > errors:
                r = errors
            else:
                r = print_code
            msg = '{0:^27s}{1:^115s}{2:^115s}'.format('Filename', 'Predicted Class', 'True Class')
            print_in_color(msg, (0, 255, 0), (55, 65, 80))
            for i in range(r):
                msg = '{0:^27s}{1:^115s}{2:^115s}'.format(error_list[i], pred_class[i], true_class[i])
                print_in_color(msg, (255, 255, 255), (55, 65, 60))
                # print(error_list[i]  , pred_class[i], true_class[i], prob_list[i])

            msg = 'Photos with only accepted error: {} \nPhotos with errors: {}'.format(total_correct, total_with_error)
            print_in_color(msg, (0, 255, 0), (55, 65, 80))
        else:
            msg = 'With accuracy of 100 % there are no errors to print'
            print_in_color(msg, (0, 255, 0), (55, 65, 80))
#Comenteaza for-urile astea pt a nu mai afisa pozele
    # for img in samples_correct:
    #     print_image(img.get('photo'), img.get('val_true'), img.get('val_pred'), good=True)
    #
    # for img in samples_with_error:
    #     print_image(img.get('photo'), img.get('val_true'), img.get('val_pred'), good=False)


def print_image(filepath, landmarks, predicted_landmarks, good):
    img = cv2.imread(
        r'C:\Users\Cera\PycharmProjects\pythonProject\celeb_faces\img_align_celeba\img_align_celeba\{path}'.format(path=filepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def transform_landmarks(landmark):
        landmarks_to_return = []
        for i in range(int(len(landmark)/2)):
            landmarks_to_return.append([landmark[i * 2], landmark[i * 2 + 1]])

        return np.array(landmarks_to_return)

    # landmarks = np.array([[69, 109], [106, 113], [77, 142], [73, 152], [108, 154]])
    landmarks = transform_landmarks(landmarks)
    predicted_landmarks = transform_landmarks(predicted_landmarks)
    #TODO:titlu = path + good
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(filepath + " " + str(good), fontsize=16)
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img)

    # ax = fig.add_subplot(1, 3, 2)
    # ax.scatter(landmarks[:, 0], -landmarks[:, 1], alpha=0.8)

    ax = fig.add_subplot(1, 2, 2)
    img2 = img.copy()

    for p in landmarks:
        img2[p[1] - 3:p[1] + 3, p[0] - 3:p[0] + 3, :] = (0, 0, 255)

    for p in predicted_landmarks:
        img2[p[1] - 3:p[1] + 3, p[0] - 3:p[0] + 3, :] = (255, 0, 0)

    ax.imshow(img2)
    plt.show()

