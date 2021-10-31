import matplotlib.pyplot as plt
import numpy as np


def show_training_samples(gen):
    images, labels = next(gen)
    plt.figure(figsize=(20, 20))
    length = len(labels)
    new_dict = {
        0: '0-12',
        1: '12-18',
        2: '18-40',
        3: '40-65',
        4: '65+',
    }
    if length < 25:
        r = length
    else:
        r = 25
    for i in range(r):
        plt.subplot(5, 5, i + 1)
        image = (images[i] + 1) / 2
        plt.imshow(image)
        class_name = np.argmax(labels[i])
        plt.title('{}: {}'.format(class_name, new_dict[class_name]), color='black', fontsize=10)
        plt.axis('off')
    plt.show()
