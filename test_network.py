import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load your trained model
model = keras.models.load_model("quadratic_solver_tf.keras")

# Example equation to test: ax^2 + bx + c = 0
a = 1.0
b = -1.0
c = -1.0

# Normalize inputs
b_over_a = b / a
c_over_a = c / a

X_test = np.array([[b_over_a, c_over_a]], dtype="float32")

# Run model
xs_pred, logd_pred, imag_logit = model.predict(X_test)

xs = xs_pred[0, 0]
d = np.exp(logd_pred[0, 0])     # undo log transform
imag_prob = tf.sigmoid(imag_logit)[0, 0].numpy()
imag= 1 if imag_prob>0.5 else 0
print("Actual mean x_S:",-b_over_a/2)
print("Predicted mean x_s:", xs)
print("Actual d:", np.sqrt(np.abs((b_over_a/2)**2-c_over_a)))
print("Predicted d:", d)
print("Real roots:", b**2-4*a*c>=0)
print("Probability imaginary:", imag_prob)
print("Predicted roots:",xs+(1j* imag+(1-imag))*d,xs-(1j*imag+1-imag)*d)