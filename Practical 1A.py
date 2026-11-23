import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report

print("TensorFlow:", tf.__version__, "| Keras:", keras.__version__)
tf.random.set_seed(42); np.random.seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("Train:", x_train.shape, y_train.shape, "| Test:", x_test.shape, y_test.shape)

x_train = (x_train / 255.0).astype("float32")[..., np.newaxis]
x_test = (x_test / 255.0).astype("float32")[..., np.newaxis]

fig, ax = plt.subplots(1, 8, figsize=(12, 2))
for i in range(8):
    ax[i].imshow(x_train[i].squeeze(), cmap="gray"); ax[i].set_title(y_train[i]); ax[i].axis("off")
plt.suptitle("Sample training images"); plt.show()

cnn = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
], name="mnist_cnn")
cnn.summary()

cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = cnn.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=2)

test_loss, test_acc = cnn.evaluate(x_test, y_test, verbose=0)
print(f"CNN test loss: {test_loss:.4f} | CNN test accuracy: {test_acc:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(history.history["loss"], "o-", label="train")
ax[0].plot(history.history["val_loss"], "o-", label="validation")
ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(history.history["accuracy"], "o-", label="train")
ax[1].plot(history.history["val_accuracy"], "o-", label="validation")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout(); plt.show()

predictions = cnn.predict(x_test, verbose=0)
pred_labels = predictions.argmax(axis=1)

fig, ax = plt.subplots(1, 8, figsize=(13, 2.2))
for i in range(8):
    ax[i].imshow(x_test[i].squeeze(), cmap="gray")
    ax[i].set_title(f"P:{pred_labels[i]} / A:{y_test[i]}", fontsize=9); ax[i].axis("off")
plt.suptitle("CNN predictions (P = predicted, A = actual)"); plt.show()

print(classification_report(y_test, pred_labels, digits=4))
cm = confusion_matrix(y_test, pred_labels)
plt.figure(figsize=(5.5, 4.5)); plt.imshow(cm, cmap="Blues"); plt.colorbar()
plt.xticks(range(10)); plt.yticks(range(10)); plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("CNN confusion matrix"); plt.show()

x_train_seq = x_train.squeeze(-1)
x_test_seq = x_test.squeeze(-1)

rnn = keras.Sequential([
    keras.Input(shape=(28, 28)),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
], name="mnist_lstm")
rnn.summary()

rnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
rnn_history = rnn.fit(x_train_seq, y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=2)

rnn_loss, rnn_acc = rnn.evaluate(x_test_seq, y_test, verbose=0)
print(f"LSTM test accuracy: {rnn_acc:.4f}")
print(f"CNN test accuracy: {test_acc:.4f}")
print("CNN parameters :", cnn.count_params(), "| LSTM parameters:", rnn.count_params())
