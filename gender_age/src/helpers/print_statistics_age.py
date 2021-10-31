
import matplotlib.pyplot as plt

def count_label(labels_list, label_to_count):
    count = 0
    for val in labels_list:
        if val[label_to_count] == 1:
            count += 1

    return count


def print_statistics(train_labels, test_labels, valid_labels):
    val_0_18 = count_label(train_labels, 0) + count_label(test_labels, 0) + count_label(valid_labels, 0)
    val_18_35 = count_label(train_labels, 1) + count_label(test_labels, 1) + count_label(valid_labels, 1)
    val_35_65 = count_label(train_labels, 2) + count_label(test_labels, 2) + count_label(valid_labels, 2)
    val_65 = count_label(train_labels, 3) + count_label(test_labels, 3) + count_label(valid_labels, 3)

    data = {'0-18': val_0_18,'18-35': val_18_35,'35-65': val_35_65, '65+': val_65 }
    courses = list(data.keys())
    values = list(data.values())

    f = plt.figure()
    plt.style.use('ggplot')
    f.set_figwidth(10)
    f.set_figheight(6)
    f.tight_layout()

    barlist = plt.bar(courses, values, color='#F7DC6F',
            width=0.4)
    barlist[0].set_color('#D7BDE2')
    barlist[1].set_color('#ABEBC6')
    barlist[2].set_color('#AED6F1')

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
    plt.title("Distribuția categoriilor de vârstă", fontsize=13, color='#424949', fontweight="bold")
    plt.show()
    # plt.savefig(newpath + '.png')