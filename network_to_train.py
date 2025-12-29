import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------- 1) data generation ----------

def generate_quadratic_data(n_samples: int):
    """
    Generate random (p,q), compute inputs (p,q) and targets:
    x_s (mean of roots), log_d, imag_flag.
    """

    # sample p,q uniformly in some range
    p = np.random.uniform(-5.0, 5.0, size=(n_samples,))
    q = np.random.uniform(-5.0, 5.0, size=(n_samples,))
    # compute normalized inputs
    X = np.stack([p, q], axis=1).astype("float32")  # shape (n, 2)

    # compute targets in double precision for accuracy
    p64 = p.astype(np.float64)
    q64 = q.astype(np.float64)

    Delta = p64 * p64 - 4.0 * q64
    xs = -p64 / (2.0 )
    d = np.sqrt(np.abs(Delta)) / (2.0)

    imag = (Delta < 0.0).astype(np.float32)  # 1 if complex, 0 if real

    eps = 1e-12  # avoid log(0)
    logd = np.log(d.astype(np.float32) + eps).astype("float32")

    y_xs = xs.astype("float32")
    y_logd = logd
    y_imag = imag

    return X, y_xs, y_logd, y_imag


# ---------- 2) build the model ----------

def build_model(hidden_width: int = 64) -> keras.Model:
    # input is just [p, q]
    inp = keras.Input(shape=(2,), name="coeffs")

    x = layers.Dense(hidden_width, activation="relu")(inp)
    x = layers.Dense(hidden_width, activation="relu")(x)
    x = layers.Dense(hidden_width, activation="relu")(x)
    x = layers.Dense(hidden_width, activation="relu")(x)
    x = layers.Dense(hidden_width, activation="relu")(x)

    # three heads:
    xs_out = layers.Dense(1, name="xs")(x)          # regression
    logd_out = layers.Dense(1, name="logd")(x)      # regression on log(d)
    imag_logit = layers.Dense(1, name="imag")(x)    # classification logit

    model = keras.Model(
        inputs=inp,
        outputs=[xs_out, logd_out, imag_logit],
        name="quadratic_solver"
    )

    # losses for each head
    loss_dict = {
        "xs": "mse",
        "logd": "mse",
        "imag": keras.losses.BinaryCrossentropy(from_logits=True),
    }

    # how much each loss contributes to the total
    loss_weights = {
        "xs": 1.0,
        "logd": 1.0,
        "imag": 0.5,
    }

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=loss_dict,
        loss_weights=loss_weights,
        metrics={"imag": [keras.metrics.BinaryAccuracy(threshold=0.0)]},
    )

    return model


# ---------- 3) training ----------

def main():
    # check GPU
    print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))

    # generate training and validation data
    N_train = 200_000
    N_val = 20_000

    X_train, y_xs_train, y_logd_train, y_imag_train = generate_quadratic_data(N_train)
    X_val, y_xs_val, y_logd_val, y_imag_val = generate_quadratic_data(N_val)

    # build model
    model = build_model(hidden_width=248)
    model.summary()
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    # train
    history = model.fit(
        X_train,
        {
            "xs": y_xs_train,
            "logd": y_logd_train,
            "imag": y_imag_train,
        },
        validation_data=(
            X_val,
            {
                "xs": y_xs_val,
                "logd": y_logd_val,
                "imag": y_imag_val,
            },
        ),
        batch_size=1024,
        epochs=200,
        callbacks=callbacks
    )

    # save model
    model.save("quadratic_solver_tf.keras")

    # quick test on a specific quadratic, say: x^2 - 3x + 2 = 0 (roots 1 and 2)
    a, b, c = 1.0, 0, 1.0
    b_over_a = b / a
    c_over_a = c / a

    X_test = np.array([[b_over_a, c_over_a]], dtype="float32")
    xs_pred, logd_pred, imag_logit_pred = model.predict(X_test)

    xs_pred = xs_pred[0, 0]
    d_pred = np.exp(logd_pred[0, 0])
    imag_prob = tf.sigmoid(imag_logit_pred)[0, 0].numpy()

    print("\nTest equation: x^2 - 3x + 2 = 0")
    print("True roots: 1 and 2")
    print("Pred mean xs (should be 1.5):", xs_pred)
    print("Pred d (should be 0.5):", d_pred)
    print("Pred imag probability (should be ~0 for real roots):", imag_prob)


if __name__ == "__main__":
    main()
