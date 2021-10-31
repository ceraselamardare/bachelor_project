import matplotlib.pyplot as plt
import numpy as np


def show_training_samples(gen):
    images, labels = next(gen)
    plt.figure(figsize=(16, 16))
    plt.style.use('ggplot')
    plt.suptitle("Portrete din baza de date",y=0.95, fontsize=16, color='#424949', fontweight="bold")

    length = len(labels)
    new_dict = {
        0: 'bald',
        1: 'black_hair',
        2: 'blond_hair',
        3: 'brown_hair',
        4: 'gray_hair',
    }
    if length < 16:
        r = length
    else:
        r = 16
    for i in range(r):
        plt.subplot(4, 4, i + 1)
        image = (images[i] + 1) / 2
        plt.imshow(image)
        class_name = np.argmax(labels[i])
        # plt.title('{}: {}'.format(class_name, new_dict[class_name]), color='black', fontsize=10)
        plt.axis('off')
    plt.show()
