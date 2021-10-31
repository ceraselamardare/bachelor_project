import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

def calc_angle(points, im):
    size = im.shape

    BA_x = (points[2][0]-points[3][0])
    BA_y = (points[2][1]-points[3][1])
    BC_x = (points[4][0]-points[3][0])
    BC_y = (points[4][1]-points[3][1])

    dot = (BA_x*BC_x) + (BA_y*BC_y)
    scalar = (BC_x*BC_x)+(BC_y*BC_y)
    r = dot/scalar
    r_x = r* BC_x
    r_y = r* BC_y
    E_x = points[3][0] + r_x
    E_y = points[3][1] + r_y

    D_x = 2*E_x - points[2][0]
    D_y = 2*E_y - points[2][1]

    image_points = np.array([
        (points[2][0], points[2][1]),  # Nose tip
        (D_x, D_y),  # Chin
        (points[0][0], points[0][1]),  # Left eye left corner
        (points[1][0], points[1][1]),  # Right eye right corne
        (points[3][0], points[3][1]),  # Left Mouth corner
        (points[4][0], points[4][1])  # Right mouth corner
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner

    ])

    # Camera internals

    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)

    points_thick = round(0.015*size[0])
    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), points_thick, (255, 255, 0), -1)

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    # p1_nou = (int(image_points[0][0]) +80, int(image_points[0][1])+80)
    # p1_nou = (int(nose_end_point2D[0][0][0]) + 20, int(nose_end_point2D[0][0][1]) + 20)

    cv2.line(im, p1, p2, (255, 255, 0), 6, cv2.LINE_AA)

    try:
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        ang1 = int(math.degrees(math.atan(m)))
    except:
        ang1 = 90
    font = cv2.FONT_HERSHEY_DUPLEX
    # cv2.putText(im, str(ang1), tuple(p1_nou), font, 2, (255, 255, 0), 2)

    return im, ang1


def plot_landmarks_angle(img_path, points_landmarks):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # img = cv2.resize(img, (712, 1200), interpolation=cv2.INTER_CUBIC)

    def transform_landmarks(landmark):
        landmarks_to_return = []
        for i in range(int(landmark.size/2)):
            landmarks_to_return.append([landmark[i * 2], landmark[i * 2 + 1]])

        return np.array(landmarks_to_return)

    points_landmarks = transform_landmarks(points_landmarks)

    angle_im, angle_text = calc_angle(points_landmarks, img.copy())


    #plot
    fig = plt.figure(figsize=(30, 15))
    ax = fig.add_subplot(1, 2, 1)
    ax.axis('off')
    ax.imshow(img)

    ax = fig.add_subplot(1, 2, 2)
    ax.axis('off')

    ax.imshow(angle_im)

    # plt.show()
    plt.savefig(r'C:\Users\Cera\PycharmProjects\pythonProject\gender_age\GUI\new_img.png', dpi = 300)
    return angle_text