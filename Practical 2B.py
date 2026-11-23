import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

tf.random.set_seed(42); np.random.seed(42)

ratings = np.array([[0, 0, 5.0], [0, 1, 3.0], [1, 0, 4.0], [1, 2, 2.0], [2, 1, 4.0], [2, 2, 5.0]])
num_users = int(ratings[:, 0].max()) + 1
num_items = int(ratings[:, 1].max()) + 1
embedding_size = 2

user_input = keras.Input(shape=(1,), name="user")
item_input = keras.Input(shape=(1,), name="item")
user_embedding = layers.Embedding(num_users, embedding_size)(user_input)
item_embedding = layers.Embedding(num_items, embedding_size)(item_input)
dot_product = layers.Flatten()(layers.Dot(axes=2)([user_embedding, item_embedding]))
toy = keras.Model([user_input, item_input], dot_product)
toy.compile(optimizer=keras.optimizers.Adam(0.05), loss="mse")

X_users, X_items = ratings[:, 0].astype("int32"), ratings[:, 1].astype("int32")
y_ratings = ratings[:, 2].astype("float32")
h = toy.fit([X_users, X_items], y_ratings, epochs=500, verbose=0)
print("Final training MSE:", round(h.history["loss"][-1], 4))

pred = toy.predict([X_users, X_items], verbose=0).ravel()
print("\nKnown ratings vs fitted:")
for u, i, r, p in zip(X_users, X_items, y_ratings, pred):
    print(f"  user {u} - item {i}: actual {r:.0f} predicted {p:.2f}")

unrated = [(0, 2), (1, 1), (2, 0)]
unrated_pred = toy.predict([np.array([u for u, _ in unrated]), np.array([i for _, i in unrated])],
                           verbose=0).ravel().clip(1, 5)
print("\nPredictions for unrated pairs:")
for (u, i), p in zip(unrated, unrated_pred):
    print(f"  user {u} - item {i}: predicted {p:.2f}")

path = "u.data"
url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
df = pd.read_csv(path if os.path.exists(path) else url, sep="\t",
                 names=["user_id", "item_id", "rating", "timestamp"])
print(df.head()); print("Shape:", df.shape)
print("Rating distribution:\n", df["rating"].value_counts().sort_index())

user_ids, item_ids = df["user_id"].unique(), df["item_id"].unique()
user2idx = {u: i for i, u in enumerate(user_ids)}
item2idx = {m: i for i, m in enumerate(item_ids)}
idx2item = {i: m for m, i in item2idx.items()}
df["user"] = df["user_id"].map(user2idx)
df["item"] = df["item_id"].map(item2idx)
num_users, num_items = len(user2idx), len(item2idx)
print("Users:", num_users, "| Items:", num_items,
      f"| Sparsity: {1 - len(df) / (num_users * num_items):.4f}")

train, test = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train, test_size=0.1, random_state=42)
X_train, y_train = [train_df["user"].values, train_df["item"].values], train_df["rating"].values.astype("float32")
X_val, y_val = [val_df["user"].values, val_df["item"].values], val_df["rating"].values.astype("float32")
X_test, y_test = [test["user"].values, test["item"].values], test["rating"].values.astype("float32")
print("Train:", len(train_df), "Validation:", len(val_df), "Test:", len(test))

def build_model(embedding_size=32, l2=1e-4, with_bias=True):
    u_in, i_in = keras.Input(shape=(1,), name="user"), keras.Input(shape=(1,), name="item")
    reg = keras.regularizers.l2(l2)
    u_vec = layers.Flatten()(layers.Embedding(num_users, embedding_size, embeddings_initializer="he_normal",
                                              embeddings_regularizer=reg)(u_in))
    i_vec = layers.Flatten()(layers.Embedding(num_items, embedding_size, embeddings_initializer="he_normal",
                                              embeddings_regularizer=reg)(i_in))
    out = layers.Dot(axes=1)([u_vec, i_vec])
    if with_bias:
        u_b = layers.Flatten()(layers.Embedding(num_users, 1)(u_in))
        i_b = layers.Flatten()(layers.Embedding(num_items, 1)(i_in))
        out = layers.Add()([out, u_b, i_b])
    out = layers.Lambda(lambda t: t + float(y_train.mean()))(out)
    return keras.Model([u_in, i_in], out)

model = build_model()
model.summary()
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=[keras.metrics.RootMeanSquaredError(name="rmse")])

early = keras.callbacks.EarlyStopping(monitor="val_rmse", patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=1024,
                    callbacks=[early], verbose=2)

plt.figure(figsize=(8, 4.5))
plt.plot(history.history["rmse"], label="Training RMSE")
plt.plot(history.history["val_rmse"], label="Validation RMSE")
plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.title("Training vs Validation RMSE"); plt.legend(); plt.show()

pred_test = model.predict(X_test, batch_size=4096, verbose=0).ravel().clip(1, 5)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))
baseline = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_train.mean())))
print(f"Test RMSE (embedding model): {rmse:.4f}")
print(f"Test RMSE (predict global mean {y_train.mean():.2f}): {baseline:.4f}")
print(f"Test MAE: {np.abs(y_test - pred_test).mean():.4f}")

print("\nSample predictions:")
print(pd.DataFrame({"user_id": test["user_id"].values[:8], "item_id": test["item_id"].values[:8],
                    "actual": y_test[:8], "predicted": pred_test[:8].round(2)}).to_string(index=False))

def recommend(user_id, n=10):
    u = user2idx[user_id]
    seen = set(df.loc[df["user"] == u, "item"])
    candidates = np.array([i for i in range(num_items) if i not in seen])
    scores = model.predict([np.full(len(candidates), u), candidates], batch_size=4096, verbose=0).ravel()
    top = candidates[np.argsort(-scores)[:n]]
    return pd.DataFrame({"item_id": [idx2item[i] for i in top],
                         "predicted_rating": np.sort(scores)[::-1][:n].round(2)})

print("User 196 rated", (df.user_id == 196).sum(), "movies. Top-10 recommendations:")
print(recommend(196).to_string(index=False))

emb_layers = [l for l in model.layers if isinstance(l, layers.Embedding)]
item_vectors = emb_layers[1].get_weights()[0]
norm = item_vectors / (np.linalg.norm(item_vectors, axis=1, keepdims=True) + 1e-9)
target = item2idx[50]
sims = norm @ norm[target]
nearest = np.argsort(-sims)[1:6]
print("Items most similar to item_id 50:", [(int(idx2item[i]), round(float(sims[i]), 3)) for i in nearest])
