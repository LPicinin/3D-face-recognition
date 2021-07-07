import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import Callback


# https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
# 192x2 = x y
#
# 192x1 = z
#
# 192x128 = descritores
# testar x, y, z + H

class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """

    def __init__(self, monitor='accuracy', baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True


# https://www.tensorflow.org/tutorials/images/cnn
# https://medium.com/analytics-vidhya/face-recognition-with-vgg-face-in-keras-96e6bc1951d5
# https://gist.github.com/EncodeTS/6bbe8cb8bebad7a672f0d872561782d9
# https://www.robots.ox.ac.uk/~vgg/software/vgg_face/
class Net:
    def __init__(self):
        xy = Input(shape=(275, 2), name="xy")
        z = Input(shape=(275, 1), name="z")
        descritores = Input(shape=(275, 128), name="descritores")
        # imagem = Input(shape=(512, 512, 3), name="Imagem")

        # modelos separados
        m_xy = Dense(275, activation="relu")(xy)
        m_xy = Dense(250, activation="relu")(m_xy)
        m_xy = Dense(198, activation="softmax")(m_xy)
        m_xy = Flatten()(m_xy)
        m_xy = Dense(500, activation="relu")(m_xy)
        self.m_xy = Model(inputs=xy, outputs=m_xy)

        m_z = Dense(275, activation="relu")(z)
        m_z = BatchNormalization()(m_z)
        m_z = Dense(350, activation="relu")(m_z)
        m_z = Dropout(0.3)(m_z)
        m_z = Dense(300, activation="relu")(m_z)
        m_z = MaxPooling1D()(m_z)
        m_z = Dense(250, activation="relu")(m_z)
        m_z = Dense(198, activation="softmax")(m_z)
        m_z = Flatten()(m_z)
        m_z = Dense(1000, activation="relu")(m_z)
        self.m_z = Model(inputs=z, outputs=m_z)

        m_descritores = Dense(1500, activation="relu")(descritores)
        m_descritores = BatchNormalization()(m_descritores)
        m_descritores = MaxPooling1D()(m_descritores)
        m_descritores = Dense(1000, activation="relu")(m_descritores)
        m_descritores = Dropout(0.4)(m_descritores)
        m_descritores = Dense(500, activation="relu")(m_descritores)
        m_descritores = Dense(400, activation="relu")(m_descritores)
        m_descritores = Dropout(0.2)(m_descritores)
        m_descritores = Dense(250, activation="relu")(m_descritores)
        m_descritores = Dense(198, activation="softmax")(m_descritores)
        m_descritores = Flatten()(m_descritores)
        m_descritores = Dense(2000, activation="relu")(m_descritores)
        self.m_descritores = Model(inputs=descritores, outputs=m_descritores)

        # m_imagem = Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(imagem)
        # m_imagem = Conv2D(16, 3, activation="relu")(m_imagem)
        # m_imagem = MaxPool2D(3)(m_imagem)
        # m_imagem = Flatten()(m_imagem)
        # self.m_imagem = Model(inputs=imagem, outputs=m_imagem)

        # combina a saída das brenchs
        self.combined = concatenate([self.m_xy.output, self.m_z.output, self.m_descritores.output], axis=-1)

        # aplique uma camada FC e, em seguida, uma previsão de regressão nas saidas combinadas

        out = Dense(1500, activation="relu")(self.combined)
        out = BatchNormalization()(out)
        # out = LayerNormalization()(out)
        out = Dense(units=1250, activation='relu')(out)
        out = Dropout(0.3)(out)
        out = Dense(units=1000, activation='relu')(out)
        out = Dense(units=500, activation='relu')(out)
        out = Dense(units=250, activation='relu')(out)
        # out = Dropout(0.5)(out)
        out = Dense(units=198, activation='softmax')(out)  # 181

        # modelo aceitará as entradas das benchs e em seguida, produz um único valor

        self.model = Model(inputs=[self.m_xy.input,
                                   self.m_z.input,
                                   self.m_descritores.input],
                           outputs=out)
        # opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        # https://keras.io/api/optimizers/adadelta/
        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
        # mtr = tf.keras.metrics.SparseCategoricalCrossentropy()
        mtr = tf.keras.metrics.SparseCategoricalAccuracy()
        los = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        # los = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # opt = tf.keras.optimizers.SGD(learning_rate=0.001)
        # los = tf.keras.losses.MSE()
        # mtr = tf.keras.metrics.CategoricalAccuracy()

        self.model.compile(loss=los, optimizer=opt, metrics=[mtr])

        # print(str(self.model.summary()))
        # print(self.model.summary())

    def fit(self, x_train, y_train, epochs, batch_size=10, callbacks=[], validation_data=None):
        return self.model.fit(x_train, y_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=True,
                              callbacks=callbacks,
                              use_multiprocessing=True, validation_data=validation_data)

    def avaliar(self, x_test, y_test, batch_size=10):
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, batch_size=batch_size)
        return test_loss, test_accuracy

    def predict(self, x, callbacks=[]):
        return self.model.predict(x, callbacks=callbacks)

    def getModel(self):
        return self.model
