import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.models import Model

from gender_age.src.helpers.print_in_color import print_in_color


class Models:
    def make_model(self, model_type, neurons_a, class_count, width, height, bands, lr, freeze, dropout, metrics):
        self.model_type = model_type
        self.class_count = class_count
        self.width = width
        self.height = height
        self.bands = bands
        self.lr = lr
        self.freeze = freeze
        self.dropout = dropout
        self.metrics = metrics
        self.neurons_a = neurons_a
        img_shape = (self.width, self.height, self.bands)
        model_list = ['Mobilenet', 'MobilenetV2', 'VGG19', 'InceptionV3', 'ResNet50V2', 'NASNetMobile', 'DenseNet201']
        if self.model_type not in model_list:
            msg = f'ERROR Modelul ales {self.model_type} nu face parte din lista de modele permise.'
            print_in_color(msg, (255, 0, 0), (55, 65, 80))
            return None
        if self.model_type == 'Mobilenet':
            base_model = tf.keras.applications.mobilenet.MobileNet(include_top=False, input_shape=img_shape,
                                                                   pooling='max', weights='imagenet', dropout=.4)
        elif self.model_type == 'MobilenetV2':
            base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=img_shape, pooling='max',
                                                           weights='imagenet')
        elif self.model_type == 'VGG19':
            base_model = tf.keras.applications.VGG19(include_top=False, input_shape=img_shape, pooling='max',
                                                     weights='imagenet')
        elif self.model_type == 'InceptionV3':
            base_model = tf.keras.applications.InceptionV3(include_top=False, input_shape=img_shape, pooling='max',
                                                           weights='imagenet')
        elif self.model_type == 'NASNetMobile':
            base_model = tf.keras.applications.NASNetMobile(include_top=False, input_shape=img_shape, pooling='max',
                                                            weights='imagenet')
        elif self.model_type == 'DenseNet201':
            base_model = tf.keras.applications.densenet.DenseNet201(include_top=False, input_shape=img_shape,
                                                                    pooling='max', weights='imagenet')
        else:
            base_model = tf.keras.applications.ResNet50V2(include_top=False, input_shape=img_shape, pooling='max',
                                                          weights='imagenet')

        if self.freeze:
            for layer in base_model.layers:
                layer.trainable = False

        x = base_model.output
        x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
        x = Dense(self.neurons_a, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=123))(x)
        # x = Dense(self.neurons_a, kernel_regularizer=regularizers.l2(l=0.016),
        #           activity_regularizer=regularizers.l1(0.006),
        #           bias_regularizer=regularizers.l1(0.006), activation='relu',
        #           kernel_initializer=tf.keras.initializers.GlorotUniform(seed=123))(x)
        x = Dropout(rate=dropout, seed=123)(x)
        output = Dense(self.class_count, activation='softmax',
                       kernel_initializer=tf.keras.initializers.GlorotUniform(seed=123))(x)
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(Adamax(lr=self.lr), loss='categorical_crossentropy', metrics=self.metrics)
        return model
