import tensorflow as tf
from tensorflow import keras

import numpy as np

from gender_age.src.helpers.print_in_color import print_in_color


# custom callback
class LRA(keras.callbacks.Callback):
    def __init__(self, model, patience, stop_patience, threshold, factor, dwell, model_name, freeze, end_epoch):
        super(LRA, self).__init__()
        self.model = model
        self.patience = patience
        self.stop_patience = stop_patience
        self.threshold = threshold
        self.factor = factor
        self.dwell = dwell
        self.lr = float(
            tf.keras.backend.get_value(model.optimizer.lr))
        self.min_tracc = 1000
        self.lowest_vloss = np.inf
        self.count = 0
        self.stop_count = 0
        self.end_epoch = end_epoch
        best_weights = self.model.get_weights()  # set a class vaiable so weights can be loaded after training is completed
        if freeze == True:
            msgs = f' Starting training using  base model {model_name} with weights frozen to imagenet weights initializing LRA callback'
        else:
            msgs = f' Starting training using base model {model_name} training all layers '
        print_in_color(msgs, (244, 252, 3), (55, 65, 80))

    def on_epoch_begin(self, epoch, logs=None):
        if epoch != 0:
            msgs = f'for epoch {epoch} '
            msgs = msgs + LRA.msg
            print_in_color(msgs, (255, 255, 0), (55, 65, 80))

    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        v_loss = logs.get('val_loss')
        mae = logs.get('mae')
        if mae > self.threshold:
            if mae < self.min_tracc:
                LRA.msg = f' training mae improved from  {self.min_tracc:7.4f} to {mae:7.4f} learning rate held at {lr:10.8f}'
                self.min_tracc = mae
                LRA.best_weights = self.model.get_weights()
                self.count = 0  # set count to 0 since training accuracy improved
                self.stop_count = 0  # set stop counter to 0
                if v_loss < self.lowest_vloss:
                    self.lowest_vloss = v_loss
            else:
                if self.count >= self.patience - 1:
                    self.lr = lr * self.factor
                    tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)
                    self.count = 0
                    self.stop_count = self.stop_count + 1
                    if self.dwell:
                        self.model.set_weights(LRA.best_weights)
                    else:
                        if v_loss < self.lowest_vloss:
                            self.lowest_vloss = v_loss
                    msgs = f' training mae {mae:7.4f} > min mae of {self.min_tracc:7.4f} '
                    LRA.msg = msgs + f' for {self.patience} epochs, lr adjusted to {self.lr:10.8f}'
                else:
                    self.count = self.count + 1
                    LRA.msg = f' training mae {mae:7.4f} < min mae of {self.min_tracc:7.4f} '
        else:
            if v_loss < self.lowest_vloss:
                msgs = f' validation loss improved from {self.lowest_vloss:8.5f} to {v_loss:8.5}, saving best weights'
                LRA.msg = msgs + f' learning rate held at {self.lr:10.8f}'
                self.lowest_vloss = v_loss
                LRA.best_weights = self.model.get_weights()
                self.count = 0
                self.stop_count = 0
            else:
                if self.count >= self.patience - 1:
                    self.lr = self.lr * self.factor  # adjust the learning rate
                    self.stop_count = self.stop_count + 1  # increment stop counter because lr was adjusted
                    msgs = f' val_loss of {v_loss:8.5f} > {self.lowest_vloss:8.5f} for {self.patience} epochs'
                    LRA.msg = msgs + f', lr adjusted to {self.lr:10.8f}'
                    self.count = 0
                    tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)
                    if self.dwell:
                        self.model.set_weights(LRA.best_weights)
                else:
                    self.count = self.count + 1
                    LRA.msg = f' validation loss of {v_loss:8.5f} > {self.lowest_vloss:8.5f}'
                if mae < self.min_tracc:
                    self.min_tracc = mae
        if epoch == self.end_epoch:
            print_in_color(LRA.msg, (255, 255, 0), (55, 65, 80))
        if self.stop_count > self.stop_patience - 1:
            LRA.msg = f' training has been halted at epoch {epoch + 1} after {self.stop_patience} adjustments of learning rate with no improvement'
            print_in_color(LRA.msg, (0, 255, 0), (55, 65, 80))
            self.model.stop_training = True
