import matplotlib.pyplot as plt
import numpy as np


def show_training_samples(gen):
    images, labels = next(gen)
    plt.figure(figsize=(20, 20))
    plt.style.use('ggplot')
    length = len(labels)
    new_dict = {
        0: 'male',
        1: 'female',

    }
    if length < 9:
        r = length
    else:
        r = 9
    for i in range(r):
        plt.subplot(3, 3, i + 1)
        image = (images[i] + 1) / 2
        plt.imshow(image)
        class_name = np.argmax(labels[i])
        # plt.title('{}: {}'.format(class_name, new_dict[class_name]), color='black', fontsize=10)
        plt.axis('off')
    plt.show()
