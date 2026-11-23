import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

tf.random.set_seed(42); np.random.seed(42)

sales = np.array([10, 12, 13, 12, 14, 15, 16, 18, 17, 19]).reshape(-1, 1)
scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(sales).flatten()

n_steps = 3
X = np.array([sales_scaled[i:i + n_steps] for i in range(len(sales_scaled) - n_steps)])
y = np.array([sales_scaled[i + n_steps] for i in range(len(sales_scaled) - n_steps)])
print("Sliding windows (scaled):\n", X.round(3), "\nTargets:", y.round(3))

mlp = keras.Sequential([keras.Input(shape=(n_steps,)), layers.Dense(8, activation="relu"), layers.Dense(1)])
mlp.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
mlp.fit(X, y, epochs=500, verbose=0)

x_input = scaler.transform(np.array([18, 17, 19]).reshape(-1, 1)).reshape(1, n_steps)
y_pred = scaler.inverse_transform(mlp.predict(x_input, verbose=0))[0][0]
print(f"Predicted sales on day 11 (from days 8-10 = 18, 17, 19): {y_pred:.2f}")

path = "daily-min-temperatures.csv"
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
df = pd.read_csv(path if os.path.exists(path) else url, parse_dates=["Date"])
print(df.head()); print("Rows:", len(df), "| from", df.Date.min().date(), "to", df.Date.max().date())
print(df["Temp"].describe().round(2))

plt.figure(figsize=(12, 3)); plt.plot(df.Date, df.Temp, lw=0.6)
plt.title("Daily minimum temperature, Melbourne"); plt.ylabel("°C"); plt.show()

data = df["Temp"].values.astype("float32")
split = int(len(data) * 0.8)
mean, std = data[:split].mean(), data[:split].std()
norm = (data - mean) / std
print(f"Train mean {mean:.3f}, std {std:.3f}")

seq_length = 14

def windows(series):
    X = np.array([series[i:i + seq_length] for i in range(len(series) - seq_length)])
    return X[..., None], series[seq_length:]

X_train, y_train = windows(norm[:split])
X_test, y_test = windows(norm[split - seq_length:])
print("X_train:", X_train.shape, "| X_test:", X_test.shape)

def build(kind):
    rnn = layers.SimpleRNN(32, activation="tanh") if kind == "SimpleRNN" else layers.LSTM(32)
    m = keras.Sequential([keras.Input(shape=(seq_length, 1)), rnn, layers.Dense(1)], name=kind)
    m.compile(optimizer="adam", loss="mse"); return m

models, preds = {}, {}
for kind in ["SimpleRNN", "LSTM"]:
    m = build(kind)
    m.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
    p = m.predict(X_test, verbose=0).ravel() * std + mean
    models[kind], preds[kind] = m, p

true = y_test * std + mean
naive = data[split - 1:-1]
print(f"{'Model':<22}{'RMSE':>8}{'MAE':>8}")
print(f"{'Naive (yesterday)':<22}{np.sqrt(mean_squared_error(true, naive)):8.3f}{mean_absolute_error(true, naive):8.3f}")
for k, p in preds.items():
    print(f"{k:<22}{np.sqrt(mean_squared_error(true, p)):8.3f}{mean_absolute_error(true, p):8.3f}")
for t, p in zip(true[:5], preds["LSTM"][:5]):
    print(f"True: {t:.2f}, Predicted (LSTM): {p:.2f}")

dates = df.Date.values[split:]
plt.figure(figsize=(12, 4.5))
plt.plot(dates, true, label="True", lw=1)
plt.plot(dates, preds["LSTM"], label="LSTM predicted", lw=1)
plt.xlabel("Date"); plt.ylabel("Temperature (°C)")
plt.title("Daily Minimum Temperature (test period): True vs Predicted"); plt.legend(); plt.show()

train_pred = models["LSTM"].predict(X_train, verbose=0).ravel() * std + mean
train_err = np.abs(y_train * std + mean - train_pred)
threshold = train_err.mean() + 3 * train_err.std()
test_err = np.abs(true - preds["LSTM"])
anom = test_err > threshold
print(f"Error threshold: {threshold:.2f} °C | anomalies found in test period: {anom.sum()} of {len(anom)} days")
print(pd.DataFrame({"Date": pd.to_datetime(dates[anom]).date, "Actual": true[anom].round(1),
                    "Predicted": preds["LSTM"][anom].round(1),
                    "Error": test_err[anom].round(1)}).to_string(index=False))

plt.figure(figsize=(12, 4))
plt.plot(dates, true, lw=0.8, label="Temperature")
plt.scatter(dates[anom], true[anom], color="red", zorder=3, label="Anomaly")
plt.title("Anomalous days detected by the LSTM forecaster"); plt.ylabel("°C"); plt.legend(); plt.show()
