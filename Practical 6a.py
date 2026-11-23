import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)
tf.random.set_seed(3); np.random.seed(3)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = (x_train / 255.0).astype("float32")[..., None]
x_test = (x_test / 255.0).astype("float32")[..., None]

src_tr, src_te = y_train < 5, y_test < 5
Xs, ys, Xs_te, ys_te = x_train[src_tr], y_train[src_tr], x_test[src_te], y_test[src_te]

rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.where(y_train == d)[0], 10, replace=False) for d in range(5, 10)])
Xt, yt = x_train[idx], y_train[idx] - 5
tgt_te = y_test >= 5
Xt_te, yt_te = x_test[tgt_te], y_test[tgt_te] - 5
print("Source train:", Xs.shape, "| Target train:", Xt.shape, "| Target test:", Xt_te.shape)

def conv_base():
    return keras.Sequential([keras.Input(shape=(28, 28, 1)),
                             layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
                             layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
                             layers.Conv2D(64, 3, activation="relu"), layers.Flatten()], name="conv_base")

def head(base, name):
    return keras.Sequential([base, layers.Dense(64, activation="relu"), layers.Dropout(0.3),
                             layers.Dense(5, activation="softmax")], name=name)

source_base = conv_base()
source_model = head(source_base, "source_model")
source_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
source_model.fit(Xs, ys, epochs=3, batch_size=128, verbose=2)
print("Source-task test accuracy (digits 0-4):", round(source_model.evaluate(Xs_te, ys_te, verbose=0)[1], 4))

results = {}

def train_target(model, name, epochs=60, lr=1e-3):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    h = model.fit(Xt, yt, epochs=epochs, batch_size=10, verbose=0, validation_data=(Xt_te, yt_te))
    acc = model.evaluate(Xt_te, yt_te, verbose=0)[1]
    results[name] = (acc, h.history["val_accuracy"]); print(f"{name:<38} target test accuracy: {acc:.4f}")

train_target(head(conv_base(), "scratch"), "From scratch (random weights)")

fe_base = conv_base(); fe_base.set_weights(source_base.get_weights()); fe_base.trainable = False
fe_model = head(fe_base, "feature_extraction")
train_target(fe_model, "Transfer: feature extraction (frozen)")
print("Trainable parameters - frozen model:", sum(np.prod(w.shape) for w in fe_model.trainable_weights),
      "of", fe_model.count_params())

fe_base.trainable = True
train_target(fe_model, "Transfer: fine-tuned (unfrozen, lr=1e-4)", epochs=30, lr=1e-4)

plt.figure(figsize=(8, 4))
for name, (acc, curve) in results.items():
    plt.plot(curve, label=f"{name} ({acc:.3f})")
plt.xlabel("Epoch"); plt.ylabel("Target test accuracy"); plt.title("Learning digits 5-9 from 50 images")
plt.legend(); plt.show()

names = list(results); accs = [results[n][0] for n in names]
plt.figure(figsize=(7, 3)); plt.barh(names, accs, color=["grey", "tab:blue", "tab:green"]); plt.xlim(0, 1)
for i, a in enumerate(accs): plt.text(a + 0.01, i, f"{a:.3f}", va="center")
plt.title("Target-task accuracy"); plt.show()

url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = pathlib.Path(keras.utils.get_file("flower_photos", origin=url, untar=True)) / "flower_photos"
IMG = (160, 160)
train_ds = keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset="training",
                                                    seed=1, image_size=IMG, batch_size=32)
val_ds = keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset="validation",
                                                  seed=1, image_size=IMG, batch_size=32)
class_names = train_ds.class_names; print(class_names)
train_ds, val_ds = train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)

base = keras.applications.MobileNetV2(input_shape=IMG + (3,), include_top=False, weights="imagenet")
base.trainable = False

inputs = keras.Input(shape=IMG + (3,))
x = layers.RandomFlip("horizontal")(inputs); x = layers.RandomRotation(0.1)(x)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x); x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
flower_model = keras.Model(inputs, outputs)
flower_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
flower_model.fit(train_ds, validation_data=val_ds, epochs=5)

base.trainable = True
for layer in base.layers[:100]: layer.trainable = False
flower_model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy",
                     metrics=["accuracy"])
flower_model.fit(train_ds, validation_data=val_ds, epochs=5)
print("Validation accuracy after fine-tuning:", flower_model.evaluate(val_ds, verbose=0)[1])
