# Create PCA encoder with orientation constraint
# Reference https://github.com/ageron/handson-ml
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
from tensorflow import keras
from auxiliary import get_data


import numpy as np

import time
from datetime import datetime
import numpy.random as rnd
import numpy as np
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

assert tf.version.VERSION == '2.3.1'
assert tf.executing_eagerly() == True

# ---------- params -------------------------------------------
output_folder = 'outputs/'

ITERATIONS = 10
# -------------------------------------------------------------


X = get_data()


class my_model(keras.Model):
    def __init__(self, units=3, X_align=None, y_align=None, **kargs):
        super().__init__(**kargs)
        self.hidden_encode = keras.layers.Dense(2, input_shape=[3])
        self.hidden_decode = keras.layers.Dense(3)
        self.X_align = X_align
        self.y_align = y_align

    def call(self, inputs):
        encoded = self.hidden_encode(inputs)
        decoded = self.hidden_decode(encoded)
        # tmp = keras.Model(inputs=inputs, outputs=encoded, name="encoder")
        return decoded

    def latent_dim(self, inputs):
        encoded = self.hidden_encode(inputs)
        return encoded
        # return keras.models.Sequential([self.hidden_encode])

    def get_encoder_model(self):
        return keras.models.Sequential([self.hidden_encode])

    # https://keras.io/guides/customizing_what_happens_in_fit/
    # https://stackoverflow.com/questions/53953099/what-is-the-purpose-of-the-tensorflow-gradient-tape
    # https://www.tensorflow.org/guide/eager

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        # from IPython.core.debugger import Tracer; Tracer()()
        # IPython.core.debugger.Pdb.set_trace()

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            y_align_pred = self.latent_dim(np.array([X_align, X_align]))
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            loss = tf.reduce_sum(tf.square(y - y_pred)) + 1 / 100 * tf.reduce_sum(
                tf.square(y_align_pred - np.array([y_align, y_align])))

        # Compute gradients
        trainable_vars = self.trainable_variables
        # print ('trainable_vars', trainable_vars)
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


X_align = [1, 1, 1]
y_align = [-0.37301758, 1.1074662]

model = my_model(units=3, X_align=X_align, y_align=y_align)
model.compile(loss="mse")

history = model.fit(X, X, epochs = 500, verbose = 0)#, callbacks = [tensorboar_cb])

model_encoder = model.get_encoder_model()


X_align_train = np.array([X_align,X_align])
y_align_train = np.array([y_align,y_align])

model_encoder.compile(loss = "mse")

coding = model_encoder.predict(X)
align_coding_vect = model_encoder.predict(X_align_train)
align_coding = align_coding_vect[0]


# ---- plot results -----------------
plt.figure()
plt.plot(coding[:,0], coding[:,1],'.')
plt.plot(y_align[0], y_align[1], 'rx')
plt.plot(align_coding[0], align_coding[1], 'r.')