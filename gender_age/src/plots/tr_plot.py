from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


def tr_plot(tr_data, start_epoch, folder='ceva'):
    tacc = tr_data.history['accuracy']
    tloss = tr_data.history['loss']
    vacc = tr_data.history['val_accuracy']
    vloss = tr_data.history['val_loss']
    epoch_count = len(tacc) + start_epoch
    epochs = []
    for i in range(start_epoch, epoch_count):
        epochs.append(i + 1)
    index_loss = np.argmin(vloss)
    val_lowest = vloss[index_loss]
    index_acc = np.argmax(vacc)
    acc_highest = vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label = 'best epoch= ' + str(index_loss + 1 + start_epoch)
    vc_label = 'best epoch= ' + str(index_acc + 1 + start_epoch)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    axes[0].plot(epochs, tloss, 'r', label='Training loss')
    axes[0].plot(epochs, vloss, 'g', label='Validation loss')
    axes[0].scatter(index_loss + 1 + start_epoch, val_lowest, s=150, c='blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(epochs, tacc, 'r', label='Training Accuracy')
    axes[1].plot(epochs, vacc, 'g', label='Validation Accuracy')
    axes[1].scatter(index_acc + 1 + start_epoch, acc_highest, s=150, c='blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout
    now = datetime.now().strftime('%Y-%m-%d_%H-%M')
    newpath = r'C:\Users\Cera\PycharmProjects\pythonProject\gender_age\rezultate\{folder}\{folder}-'.format(folder=folder) + now
    plt.savefig(newpath + '.png')
