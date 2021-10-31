import numpy as np
import cv2
from math import pi, cos, sin


def angle_data_aug(train_images, landmarks):
    ####FLIP
    def left_right_flip(image, keypoints):
        total = 0
        flipped_keypoints = []
        flipped_image = np.flip(image, axis=2)  # Flip column-wise (axis=2)
        print('Flipped {} images'.format(len(image)))
        for idx, sample_keypoints in enumerate(keypoints):
            flipped_keypoints.append([178. - coor if idx % 2 == 0 else coor for idx, coor in enumerate(
                sample_keypoints)])  # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping
            total += 1
            if total % 100 == 0:
                print('Flipped {}/{} keypoints'.format(total, len(keypoints)))
        return flipped_image, flipped_keypoints

    flipped_train_images, flipped_train_keypoints = left_right_flip(train_images, landmarks)
    # print("Shape of flipped_train_images: {}".format(np.shape(flipped_train_images)))
    # print("Shape of flipped_train_keypoints: {}".format(np.shape(flipped_train_keypoints)))
    train_images = np.concatenate((train_images, flipped_train_images))
    landmarks = np.concatenate((landmarks, flipped_train_keypoints))

    #######ROTATION
    rotation_angles = [12]

    def rotate_image(images, keypoints):
        total = 0
        total_k = 0
        rotated_images = []
        rotated_keypoints = []
        for angle_a in rotation_angles:
            for angle in [angle_a, -angle_a]:
                angle_rad = -angle * pi / 180.
                for image in images:
                    image_center = tuple(np.array(image.shape[1::-1]) / 2)
                    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
                    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
                    rotated_images.append(result)
                    total += 1
                    if total % 100 == 0:
                        print('Rotated {}/{} images'.format(total, len(images)))

                for keypoint in keypoints:
                    rotated_keypoint = keypoint - 109.  # Subtract the middle value of the image dimension
                    for idx in range(0, len(rotated_keypoint), 2):
                        # https://in.mathworks.com/matlabcentral/answers/93554-how-can-i-rotate-a-set-of-points-in-a-plane-by-a-certain-angle-about-an-arbitrary-point
                        rotated_keypoint[idx] = rotated_keypoint[idx] * cos(angle_rad) - rotated_keypoint[
                            idx + 1] * sin(
                            angle_rad)
                        rotated_keypoint[idx + 1] = rotated_keypoint[idx] * sin(angle_rad) + rotated_keypoint[
                            idx + 1] * cos(angle_rad)
                    rotated_keypoint += 109.  # Add the earlier subtracted value
                    rotated_keypoints.append(rotated_keypoint)
                    total_k += 1
                    if total_k % 100 == 0:
                        print('Rotated {}/{} images'.format(total_k, len(keypoints)))

        return rotated_images, rotated_keypoints

    img_r, land_r = rotate_image(train_images, landmarks)
    train_images = np.concatenate((train_images, img_r))
    landmarks = np.concatenate((landmarks, land_r))

    return train_images, landmarks
