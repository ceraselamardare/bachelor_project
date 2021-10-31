from gender_age.src.helpers.print_in_color import print_in_color


def display_eval_metrics(e_data, metrics=['accuracy']):
    msg = 'Model Metrics after Training'
    print_in_color(msg, (255, 255, 0), (55, 65, 80))
    msg = '{0:^24s}{1:^24s}'.format('Metric', 'Value')
    print_in_color(msg, (255, 255, 0), (55, 65, 80))
    for key, value in e_data.items():
        print(f'{key:^24s}{value:^24.5f}')
    if 'accuracy' in metrics:
        acc = e_data['accuracy'] * 100
        return acc
    else:
        for metric in metrics:
            val = e_data[metric]
        return val
