from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


def tr_plot_angle(tr_data, start_epoch, folder='ceva'):
    trse = tr_data.history['mae']
    tloss = tr_data.history['loss']
    vrse = tr_data.history['val_mae']
    vloss = tr_data.history['val_loss']
    epoch_count = len(trse) + start_epoch
    epochs = []
    for i in range(start_epoch, epoch_count):
        epochs.append(i + 1)
    index_loss = np.argmin(vloss)
    val_lowest = vloss[index_loss]
    index_acc = np.argmin(vrse)
    rse_lowest = vrse[index_acc]
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
    axes[1].plot(epochs, trse, 'r', label='Training Mean Absolute Error')
    axes[1].plot(epochs, vrse, 'g', label='Validation Mean Absolute Error')
    axes[1].scatter(index_acc + 1 + start_epoch, rse_lowest, s=150, c='blue', label=vc_label)
    axes[1].set_title('Training and Validation Mean Absolute Error')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].legend()
    plt.tight_layout
    now = datetime.now().strftime('%Y-%m-%d_%H-%M')
    newpath = r'C:\Users\Cera\PycharmProjects\pythonProject\gender_age\rezultate\{folder}\{folder}-'.format(folder=folder) + now
    plt.savefig(newpath + '.png')
