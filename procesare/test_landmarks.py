import cv2
from matplotlib import pyplot as plt
import numpy as np


img = cv2.imread(r'C:\Users\Cera\PycharmProjects\pythonProject\celeb_faces\img_align_celeba\img_align_celeba\000001.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
landmarks = np.array([[69, 109], [106, 113], [77, 142], [73, 152], [108, 154]])
# predicted_landmarks = np.array([[69, 109], [106, 113], [77, 142], [73, 152], [106, 154]])

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 2, 1)
ax.imshow(img)

# ax = fig.add_subplot(1, 3, 2)
# ax.scatter(landmarks[:, 0], -landmarks[:, 1], alpha=0.8)

ax = fig.add_subplot(1, 2, 2)
img2 = img.copy()

for p in landmarks:
    img2[p[1]-3:p[1]+3, p[0]-3:p[0]+3, :] = (255, 0, 0)

# for p in predicted_landmarks:
#     img2[p[1]-3:p[1]+3, p[0]-3:p[0]+3, :] = (0, 0, 255)

ax.imshow(img2)
plt.show()