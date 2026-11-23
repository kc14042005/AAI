import os, re, string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

print("TensorFlow:", tf.__version__)
tf.random.set_seed(42)

texts = ["I love this product", "This is the worst movie", "I am so happy",
         "I hate this book", "What a great day", "I am very disappointed"]
labels = np.array([1, 0, 1, 0, 1, 0])

vectorizer = layers.TextVectorization(output_mode="int", output_sequence_length=5)
vectorizer.adapt(texts)
print("Vocabulary:", vectorizer.get_vocabulary())
padded = vectorizer(np.array(texts))
print("Padded sequences:\n", padded.numpy())
vocab_size = vectorizer.vocabulary_size()

toy = keras.Sequential([
    keras.Input(shape=(5,)),
    layers.Embedding(input_dim=vocab_size, output_dim=8),
    layers.Flatten(),
    layers.Dense(6, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
toy.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
h = toy.fit(padded, labels, epochs=100, verbose=0)
print("Final training accuracy:", h.history["accuracy"][-1], "| loss:", round(h.history["loss"][-1], 4))

test_texts = ["I love this product", "worst movie", "I am so happy", "I hate this book",
              "great day", "I am very disappointed", "so happy", "What a worst day"]
predictions = toy.predict(vectorizer(np.array(test_texts)), verbose=0)
for text, p in zip(test_texts, predictions):
    print(f"{text:<26} -> {'Positive' if p[0] > 0.5 else 'Negative'} ({p[0]:.2f})")

csv = "IMDB-Dataset.csv"
url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
df = pd.read_csv(csv if os.path.exists(csv) else url)
df["label"] = (df["sentiment"] == "positive").astype(int)
print(df.shape); print(df["sentiment"].value_counts())
print("\nSample review:\n", df["review"][1][:400], "...")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=42, stratify=train_df["label"])
print("Train:", len(train_df), "Validation:", len(val_df), "Test:", len(test_df))

batch_size = 32

def make_ds(d, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((d["review"].values, d["label"].values))
    return (ds.shuffle(10000, seed=42) if shuffle else ds).batch(batch_size)

raw_train_ds, raw_val_ds, raw_test_ds = make_ds(train_df, True), make_ds(val_df), make_ds(test_df)

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, "[%s]" % re.escape(string.punctuation), "")

max_features, sequence_length = 10000, 250
vectorize_layer = layers.TextVectorization(standardize=custom_standardization, max_tokens=max_features,
                                           output_mode="int", output_sequence_length=sequence_length)
vectorize_layer.adapt(raw_train_ds.map(lambda x, y: x))

def vectorize_text(text, label):
    return vectorize_layer(text), label

text_batch, label_batch = next(iter(raw_train_ds))
print("Review:", text_batch[0].numpy()[:150], "...")
print("Label:", label_batch[0].numpy())
print("Vectorized:", vectorize_layer(text_batch[:1]).numpy()[0][:20], "...")
print("1287 --->", vectorize_layer.get_vocabulary()[1287], "| 313 --->", vectorize_layer.get_vocabulary()[313])
print("Vocabulary size:", len(vectorize_layer.get_vocabulary()))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = raw_train_ds.map(vectorize_text).cache().prefetch(AUTOTUNE)
val_ds = raw_val_ds.map(vectorize_text).cache().prefetch(AUTOTUNE)
test_ds = raw_test_ds.map(vectorize_text).cache().prefetch(AUTOTUNE)

embedding_dim = 16
model = keras.Sequential([
    keras.Input(shape=(sequence_length,)),
    layers.Embedding(max_features, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")])
model.summary()
model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer="adam",
              metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)])
history = model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=2)

loss, accuracy = model.evaluate(test_ds, verbose=0)
print("Test loss:", round(loss, 4), "| Test accuracy:", round(accuracy, 4))

hd = history.history
ep = range(1, len(hd["loss"]) + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(ep, hd["loss"], "bo", label="Training loss")
ax[0].plot(ep, hd["val_loss"], "b", label="Validation loss")
ax[0].set_title("Training and validation loss"); ax[0].set_xlabel("Epochs"); ax[0].legend()
ax[1].plot(ep, hd["binary_accuracy"], "bo", label="Training acc")
ax[1].plot(ep, hd["val_binary_accuracy"], "b", label="Validation acc")
ax[1].set_title("Training and validation accuracy"); ax[1].set_xlabel("Epochs"); ax[1].legend(loc="lower right")
plt.tight_layout(); plt.show()

export_model = keras.Sequential([keras.Input(shape=(1,), dtype="string"), vectorize_layer, model])
export_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
raw_test_2d = raw_test_ds.map(lambda x, y: (tf.expand_dims(x, -1), y))
print(export_model.evaluate(raw_test_2d, return_dict=True, verbose=0))

examples = tf.constant([["The movie was great!"], ["The movie was okay."], ["The movie was terrible..."],
                        ["The movie was worse..."], ["The movie was good..."],
                        ["An absolute masterpiece with brilliant acting"],
                        ["Boring, predictable and far too long"]])
scores = export_model.predict(examples, verbose=0).ravel()
for t, s in zip(examples.numpy().ravel(), scores):
    print(f"{t.decode():<48} {s:.3f} {'Positive' if s >= 0.5 else 'Negative'}")

lstm_model = keras.Sequential([
    keras.Input(shape=(sequence_length,)),
    layers.Embedding(max_features, 32, mask_zero=True),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")])
lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
lstm_model.summary()
lstm_hist = lstm_model.fit(train_ds, validation_data=val_ds, epochs=2, verbose=2)
print("BiLSTM test accuracy:", round(lstm_model.evaluate(test_ds, verbose=0)[1], 4))
