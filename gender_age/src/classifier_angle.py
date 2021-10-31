import tensorflow as tf

import os

from opt_einsum.backends import tensorflow

from gender_age.src.classes.LRA_angle import LRA
from gender_age.src.classes.Models_angle import Models
from gender_age.src.helpers.display_eval_metrics import display_eval_metrics
from gender_age.src.helpers.print_in_color import print_in_color
from gender_age.src.helpers.print_info_angle import print_info
from gender_age.src.make_gens_angle import make_gens
from gender_age.src.plots.tr_plot import tr_plot
from gender_age.src.helpers.get_images_angle import get_images
import logging
from tensorflow.python.client import device_lib

from gender_age.src.plots.tr_plot_angle import tr_plot_angle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def classifier_angle(save_dir,
                     model_type='Mobilenet',
                     total_images=1000,
                     epochs=2,
                     freeze=True,
                     height=218,
                     width=178,
                     bands=3,
                     batch_size=32,
                     lr=.0001,
                     patience=10,
                     stop_patience=50,
                     threshold=4.5,
                     dwell=True,
                     factor=.5,
                     dropout=.1,
                     print_code=10,
                     neurons_a=128,
                     metrics=['mae']):
    # metrics=[]):
    tensorflow._get_tensorflow_and_device()
    print(device_lib.list_local_devices())

    tf.random.set_seed(0)

    # generarea structurilor de date

    train_images, train_labels, train_paths, \
    valid_images, valid_labels, valid_paths, \
    test_images, test_labels, test_paths = make_gens(height, width, batch_size, total_images)

    # afisarea unor imagini de train
    # show_training_samples(train_gen)
    # print_statistics(train_gen.y, test_gen.y, valid_gen.y)

    # determinarea numarului de clase
    class_count = 10

    # creearea modelului pornind de la modelul de baza
    model = Models().make_model(model_type, neurons_a, class_count, width, height, bands, lr, freeze, dropout, metrics)

    # generarea unei functii de callback customizata pornind de la parametrii initiali
    callbacks = [LRA(model=model, patience=patience, stop_patience=stop_patience, threshold=threshold,
                     factor=factor, dwell=dwell, model_name=model_type, freeze=freeze, end_epoch=epochs - 1)]

    # antrenarea modelului

    results = model.fit(x=train_images, y=train_labels, epochs=epochs, verbose=2, callbacks=callbacks,
                        validation_data=(valid_images, valid_labels),
                        validation_steps=None, shuffle=False, initial_epoch=0)

    # plotarea pierderilor si a acuratetii
    tr_plot_angle(results, 0, 'angle')

    # pastrarea best weights pe parcursul invatarii
    model.set_weights(LRA.best_weights)

    e_dict = model.evaluate(x=test_images, y=test_labels, verbose=1, steps=None)
    e_dict = {out: e_dict[i] for i, out in enumerate(model.metrics_names)}
    rmse = display_eval_metrics(e_dict, metrics=['mae'])
    msg = f' Mean absolute error on the test set is {rmse:5.2f} %'
    print_in_color(msg, (0, 255, 0), (55, 65, 80))
    save_id = str(model_type + '-' + str(rmse)[:str(rmse).rfind('.') + 3] + '.h5')

    # salvarea modelului
    save_loc = os.path.join(save_dir, save_id)
    model.save(save_loc)

    # afisarea predictiilor
    preds = model.predict(test_images, verbose=0, steps=None)
    # print_info(test_labels, test_paths, preds, print_code)
