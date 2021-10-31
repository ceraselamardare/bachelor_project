from gender_age.src.helpers.print_in_color import print_in_color
import numpy as np

import matplotlib.pyplot as plt


def print_info(test_gen, preds, print_code):
    labels = test_gen.y
    file_names = test_gen.filenames_paths_ex
    error_list = []
    true_class = []
    pred_class = []
    prob_list = []
    new_dict = {
        0: '0-12',
        1: '12-18',
        2: '18-40',
        3: '40-65',
        4: '65+',
    }
    error_indices = []
    errors = 0
    for i, p in enumerate(preds):
        prediction = np.argmax(p)
        true_index = np.argmax(labels[i])  # labels are integer values
        if prediction != true_index:  # a misclassification has occurred
            error_list.append(file_names[i].replace('C:/Users/Cera/PycharmProjects/pythonProject/imdb_0/', ''))
            true_class.append(new_dict[true_index])
            pred_class.append(new_dict[prediction])
            prob_list.append(p[prediction])
            error_indices.append(prediction)
            errors = errors + 1
    if print_code != 0:
        if errors > 0:
            if print_code > errors:
                r = errors
            else:
                r = print_code
            msg = '{0:^50s}{1:^28s}{2:^28s}{3:^16s}'.format('Filename', 'Predicted Class', 'True Class', 'Probability')
            print_in_color(msg, (0, 255, 0), (55, 65, 80))
            for i in range(r):
                msg = '{0:^28s}{1:^28s}{2:^28s}{3:4s}{4:^6.4f}'.format(error_list[i], pred_class[i], true_class[i], ' ',
                                                                       prob_list[i])
                print_in_color(msg, (255, 255, 255), (55, 65, 60))
                # print(error_list[i]  , pred_class[i], true_class[i], prob_list[i])
        else:
            msg = 'With accuracy of 100 % there are no errors to print'
            print_in_color(msg, (0, 255, 0), (55, 65, 80))
    if errors > 0:
        plot_bar = []
        plot_class = []
        for key, value in new_dict.items():
            count = error_indices.count(key)
            if count != 0:
                plot_bar.append(count)
                plot_class.append(value)
        fig = plt.figure()
        fig.set_figheight(len(plot_class) / 3)
        fig.set_figwidth(10)
        plt.style.use('fivethirtyeight')
        for i in range(0, len(plot_class)):
            c = plot_class[i]
            x = plot_bar[i]
            plt.barh(c, x, )
            plt.title(' Errors by Class on Test Set')
        # plt.show()
