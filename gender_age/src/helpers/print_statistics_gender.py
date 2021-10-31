from gender_age.src.helpers.print_in_color import print_in_color
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns


def count_label(labels_list, label_to_count):
    count = 0
    for val in labels_list:
        if val[label_to_count] == 1:
            count += 1

    return count


def print_statistics(train_labels, test_labels, valid_labels):
    males = count_label(train_labels, 0) + count_label(test_labels, 0) + count_label(valid_labels, 0)
    females = count_label(train_labels, 1) + count_label(test_labels, 1) + count_label(valid_labels, 1)

    data = {'Bărbați': males, 'Femei': females}
    courses = list(data.keys())
    values = list(data.values())

    f = plt.figure()
    plt.style.use('ggplot')
    f.set_figwidth(6)
    f.set_figheight(6)
    f.tight_layout()
    barlist = plt.bar(courses, values, align='center', color='#F9E79F',
                      width=0.4)
    barlist[0].set_color('#85C1E9')
    for bar in barlist:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            round(bar.get_height(), 1),
            horizontalalignment='center',
            color='#5F6A6A',
            weight='bold'
        )
    plt.xlabel("Clasa")
    plt.ylabel("Numărul de imagini")

    plt.title("Distribuția bărbaților și femeilor", fontsize=13, color='#424949', fontweight="bold")

    plt.show()
    now = datetime.now().strftime('%Y-%m-%d_%H-%M')
    # newpath = r'C:\Users\Cera\PycharmProjects\pythonProject\gender_age\rezultate\gender' + now
    # plt.savefig(newpath + '.png')
