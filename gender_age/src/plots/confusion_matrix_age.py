from datetime import datetime

import numpy as np
import seaborn as sns
from tensorflow.python.ops.confusion_matrix import confusion_matrix
import matplotlib.pyplot as plt


def confusion_matrix_age(labels, preds):
    y_pred = []
    y_true = []
    classes = ['0-18', '18-35', '35-65', '65+']
    for i, p in enumerate(preds):
        pred_index = np.argmax(p)
        y_pred.append(pred_index)

    for i, l in enumerate(labels):
        label_index = np.argmax(l)
        y_true.append(label_index)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
    plt.xticks(np.arange(len(classes)) + .5, classes, rotation=90)
    plt.yticks(np.arange(len(classes)) + .5, classes, rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    now = datetime.now().strftime('%Y-%m-%d_%H-%M')
    path = r'C:\Users\Cera\PycharmProjects\pythonProject\gender_age\rezultate\age\confusion_matrix_age_{}.png'.format(now)
    plt.savefig(path)