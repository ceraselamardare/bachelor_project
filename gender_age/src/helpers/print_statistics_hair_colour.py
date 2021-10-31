
import matplotlib.pyplot as plt

def count_label(labels_list, label_to_count):
    count = 0
    for val in labels_list:
        if val[label_to_count] == 1:
            count += 1

    return count


def print_statistics(train_labels, test_labels, valid_labels):
    bald = count_label(train_labels, 0) + count_label(test_labels, 0) + count_label(valid_labels, 0)
    black_hair = count_label(train_labels, 1) + count_label(test_labels, 1) + count_label(valid_labels, 1)
    blond_hair = count_label(train_labels, 2) + count_label(test_labels, 2) + count_label(valid_labels, 2)
    brown_hair = count_label(train_labels, 3) + count_label(test_labels, 3) + count_label(valid_labels, 3)
    gray_hair = count_label(train_labels, 4) + count_label(test_labels, 4) + count_label(valid_labels, 4)

    data = {'Chel': bald, 'Păr negru': black_hair,'Păr blond': blond_hair,'Păr castaniu': brown_hair, 'Păr cărunt': gray_hair}
    courses = list(data.keys())
    values = list(data.values())

    f = plt.figure()
    plt.style.use('ggplot')
    f.set_figwidth(10)
    f.set_figheight(6)
    f.tight_layout()

    barlist = plt.bar(courses, values, color='#EC7063',
            width=0.4)
    barlist[0].set_color('#AF7AC5')
    barlist[1].set_color('#F5B041')
    barlist[2].set_color('#85C1E9')
    barlist[3].set_color('#F9E79F')

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
    plt.title("Distribuția culorilor părului", fontsize=13, color='#424949', fontweight="bold")
    plt.show()
    # plt.savefig(newpath + '.png')

