import tensorflow as tf

from tensorflow.keras.models import Model

from keras.layers import Dense, Dropout

from tensorflow.python.keras import Input

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
            msg = f'ERROR modelul ales {self.model_type} nu este un nume de model permis'
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

        image = Input(shape=(224, 224, 3))

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu')(image)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)

        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(
            x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(
            x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)

        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(
            x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(
            x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)

        x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(
            x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(
            x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)

        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(
            x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(
            x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=1024, activation='relu', use_bias=True)(x)
        x = Dropout(rate=dropout, seed=123)(x)

        output = Dense(self.class_count, activation='relu', use_bias=True, name='output')(x)
        model = Model(inputs=image, outputs=output)
        model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.keras.optimizers.Adam(lr=self.lr),
                      metrics=self.metrics)
        print(model.summary())
        return model
