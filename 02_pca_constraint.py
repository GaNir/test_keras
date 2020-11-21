# Create PCA encoder with orientation constraint
# Reference https://github.com/ageron/handson-ml

#IPython.core.debugger.Pdb.set_trace()


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
output_folder = 'output/'
# -------------------------------------------------------------
now = datetime.now()
file_header = output_folder + now.strftime("%Y%m%d_%H%M%S_")

X = get_data()

from sklearn.decomposition import PCA
X_pca = PCA(n_components=2).fit_transform(X)

loss_tracker = keras.metrics.Mean(name="loss")
loss_shift_tracker = keras.metrics.Mean(name="loss")


class my_model(keras.Model):
    def __init__(self, units=3, X_align=None, y_align=None, **kargs):
        super().__init__(**kargs)
        self.hidden_encode = keras.layers.Dense(2, input_shape=[3])
        self.hidden_decode = keras.layers.Dense(3)
        self.X_align = X_align
        self.y_align = y_align

    #         self.metrics.append(keras.metrics.Mean(name="base_loss"))
    #         self.metrics.append(keras.metrics.Mean(name="shift_loss"))

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
        x, y = data

        # from IPython.core.debugger import Tracer; Tracer()()

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            y_align_pred = self.latent_dim(np.array([X_align, X_align]))
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            shift_loss = tf.reduce_sum(tf.square(y_align_pred - np.array([y_align, y_align])))
            base_loss = tf.reduce_sum(tf.square(y - y_pred))
            loss = base_loss + 1 / x.shape[0] * shift_loss
            # l = tf.reduce_sum(tf.square(y_align_pred - np.array([y_align,y_align])))

        #             loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses) + l

        # Compute gradients
        trainable_vars = self.trainable_variables
        # print ('trainable_vars', trainable_vars)
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        loss_tracker.update_state(base_loss)
        loss_shift_tracker.update_state(shift_loss)
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        dict_res = {m.name: m.result() for m in self.metrics}
        dict_res.update({"base_loss": loss_tracker.result(), "shift_loss": loss_shift_tracker.result()})
        return dict_res


# X_align = [1,1,1]
# y_align = [-0.37301758,  1.1074662]
X_align = [0, 0, 0]
y_align = [0, 0]


def plot_model_res(model):
    model_encoder = model.get_encoder_model()
    X_align_train = np.array([X_align,X_align])
    y_align_train = np.array([y_align,y_align])
    model_encoder.compile(loss = "mse")
    coding = model_encoder.predict(X)
    align_coding_vect = model_encoder.predict(X_align_train)
    align_coding = align_coding_vect[0]
    plt.plot(coding[:,0], coding[:,1], '.', alpha = 0.2)
    plt.plot(y_align[0], y_align[1], 'rx')
    plt.plot(align_coding[0], align_coding[1], 'r.')


MODEL_NUM = 3
model_list = []
# opt = keras.optimizers.Adam(learning_rate=0.01)
opt = keras.optimizers.SGD(learning_rate=0.01)

for i in range(MODEL_NUM):
    print(f'{i} / {MODEL_NUM}       \r', end = '')
    model = my_model(units=3, X_align=X_align, y_align=y_align)
    model.compile(loss = 'mse', optimizer = opt)
    history = model.fit(X, X, epochs=20, verbose=0, batch_size=10)#, callbacks = [tensorboar_cb])
    model_list.append(model)



key_list = ['base_loss', 'shift_loss']


def save_plot(plt, filename):
    print(f'saving {filename}')
    plt.savefig(filename)


plt.figure()
for k in key_list:
    plt.plot(history.history[k])
plt.legend(key_list)

save_plot(plt, file_header+'loss.png')


plt.figure()

for model in model_list:
    plot_model_res(model)
plt.scatter(X_pca[:,0], X_pca[:,1], s=5, c='C7',alpha = 0.2)

save_plot(plt, file_header+'dim_reduction.png')


