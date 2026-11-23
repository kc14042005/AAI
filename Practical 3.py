import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__, "| OpenCV:", cv2.__version__)
rng = np.random.default_rng(0); tf.random.set_seed(0)

CLASSES = ["circle", "square", "triangle"]
IMG = 64

def draw_shape(img, mask, cls, cx, cy, r, colour):
    if cls == 0:
        cv2.circle(img, (cx, cy), r, colour, -1); cv2.circle(mask, (cx, cy), r, 1, -1)
    elif cls == 1:
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), colour, -1)
        cv2.rectangle(mask, (cx - r, cy - r), (cx + r, cy + r), 2, -1)
    else:
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
        cv2.fillPoly(img, [pts], colour); cv2.fillPoly(mask, [pts], 3)

def background():
    base = rng.integers(0, 80, 3)
    return np.clip(base + rng.normal(0, 12, (IMG, IMG, 3)), 0, 255).astype(np.uint8)

def colour():
    return tuple(int(c) for c in rng.integers(120, 256, 3))

def make_detection_data(n):
    X = np.zeros((n, IMG, IMG, 3), np.float32); cls = np.zeros(n, np.int32); box = np.zeros((n, 4), np.float32)
    for i in range(n):
        img = background().copy(); m = np.zeros((IMG, IMG), np.uint8)
        c = int(rng.integers(0, 3)); r = int(rng.integers(6, 16))
        cx, cy = int(rng.integers(r, IMG - r)), int(rng.integers(r, IMG - r))
        draw_shape(img, m, c, cx, cy, r, colour())
        X[i] = img / 255.0; cls[i] = c
        box[i] = [(cx - r) / IMG, (cy - r) / IMG, (cx + r) / IMG, (cy + r) / IMG]
    return X, cls, box

Xd_train, cd_train, bd_train = make_detection_data(4000)
Xd_test, cd_test, bd_test = make_detection_data(500)
print("Detection train:", Xd_train.shape, "| test:", Xd_test.shape)

def show_boxes(X, cls, boxes, pred_cls=None, pred_boxes=None, n=6, title=""):
    fig, ax = plt.subplots(1, n, figsize=(2.2 * n, 2.6))
    for i in range(n):
        ax[i].imshow(X[i]); ax[i].axis("off")
        x1, y1, x2, y2 = boxes[i] * IMG
        ax[i].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=2))
        t = CLASSES[cls[i]]
        if pred_boxes is not None:
            px1, py1, px2, py2 = pred_boxes[i] * IMG
            ax[i].add_patch(plt.Rectangle((px1, py1), px2 - px1, py2 - py1, fill=False, ec="red", lw=2, ls="--"))
            t = f"{CLASSES[pred_cls[i]]} (true {CLASSES[cls[i]]})"
        ax[i].set_title(t, fontsize=8)
    plt.suptitle(title); plt.show()

show_boxes(Xd_train, cd_train, bd_train, title="Training samples with ground-truth boxes")

inp = keras.Input(shape=(IMG, IMG, 3))
x = inp
for f in [32, 64, 128]:
    x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
class_out = layers.Dense(3, activation="softmax", name="class")(x)
box_out = layers.Dense(4, activation="sigmoid", name="box")(x)
detector = keras.Model(inp, [class_out, box_out], name="shape_detector")
detector.compile(optimizer="adam",
                 loss={"class": "sparse_categorical_crossentropy", "box": "mse"},
                 loss_weights={"class": 1.0, "box": 10.0},
                 metrics={"class": "accuracy"})
detector.summary()
hist = detector.fit(Xd_train, {"class": cd_train, "box": bd_train}, validation_split=0.1,
                    epochs=12, batch_size=64, verbose=2)

def iou(a, b):
    x1, y1 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda t: (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
    return inter / (area(a) + area(b) - inter)

pc, pb = detector.predict(Xd_test, verbose=0)
pc = pc.argmax(1)
ious = iou(bd_test, pb)
print(f"Classification accuracy : {(pc == cd_test).mean():.4f}")
print(f"Mean IoU of boxes : {ious.mean():.4f}")
print(f"Detections with IoU>=0.5: {(ious >= 0.5).mean():.4f}")
print(f"Correct class AND IoU>=0.5 (detection accuracy @0.5): {((pc == cd_test) & (ious >= 0.5)).mean():.4f}")
show_boxes(Xd_test, cd_test, bd_test, pc, pb, title="Test predictions (green = truth, red dashed = predicted)")

def make_seg_data(n):
    X = np.zeros((n, IMG, IMG, 3), np.float32); Y = np.zeros((n, IMG, IMG), np.uint8)
    for i in range(n):
        img = background().copy(); m = np.zeros((IMG, IMG), np.uint8)
        for _ in range(int(rng.integers(1, 4))):
            r = int(rng.integers(5, 13))
            draw_shape(img, m, int(rng.integers(0, 3)), int(rng.integers(r, IMG - r)),
                       int(rng.integers(r, IMG - r)), r, colour())
        X[i] = img / 255.0; Y[i] = m
    return X, Y

Xs_train, Ys_train = make_seg_data(3000)
Xs_test, Ys_test = make_seg_data(300)
print("Segmentation train:", Xs_train.shape, Ys_train.shape, "| classes: 0=background, 1=circle, 2=square, 3=triangle")

def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    return layers.Conv2D(f, 3, padding="same", activation="relu")(x)

def build_unet(n_classes=4):
    inp = keras.Input(shape=(IMG, IMG, 3))
    c1 = conv_block(inp, 16); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 32); p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 64); p3 = layers.MaxPooling2D()(c3)
    b = conv_block(p3, 128)
    u3 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(b)
    c4 = conv_block(layers.Concatenate()([u3, c3]), 64)
    u2 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c4)
    c5 = conv_block(layers.Concatenate()([u2, c2]), 32)
    u1 = layers.Conv2DTranspose(16, 2, strides=2, padding="same")(c5)
    c6 = conv_block(layers.Concatenate()([u1, c1]), 16)
    out = layers.Conv2D(n_classes, 1, activation="softmax")(c6)
    return keras.Model(inp, out, name="unet")

unet = build_unet()
unet.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("U-Net parameters:", unet.count_params())
seg_hist = unet.fit(Xs_train, Ys_train, validation_split=0.1, epochs=10, batch_size=32, verbose=2)

P = unet.predict(Xs_test, verbose=0).argmax(-1)
print(f"Pixel accuracy: {(P == Ys_test).mean():.4f}")
ious = []
for k in range(4):
    inter = ((P == k) & (Ys_test == k)).sum(); union = ((P == k) | (Ys_test == k)).sum()
    ious.append(inter / union)
    print(f"IoU {(['background'] + CLASSES)[k]:<10}: {inter / union:.4f}")
print(f"Mean IoU: {np.mean(ious):.4f}")

cmap = plt.matplotlib.colors.ListedColormap(["black", "tab:red", "tab:green", "tab:blue"])
fig, ax = plt.subplots(3, 5, figsize=(11, 7))
for j in range(5):
    ax[0, j].imshow(Xs_test[j]); ax[1, j].imshow(Ys_test[j], cmap=cmap, vmin=0, vmax=3)
    ax[2, j].imshow(P[j], cmap=cmap, vmin=0, vmax=3)
    for i in range(3): ax[i, j].axis("off")
ax[0, 0].set_title("Image", loc="left"); ax[1, 0].set_title("True mask", loc="left")
ax[2, 0].set_title("U-Net prediction", loc="left")
plt.suptitle("Segmentation (red = circle, green = square, blue = triangle)"); plt.tight_layout(); plt.show()

from ultralytics import YOLO
yolo = YOLO("yolo11n.pt")
results = yolo("https://ultralytics.com/images/bus.jpg")
r = results[0]
for c, conf, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy):
    print(f"{r.names[int(c)]:<10} confidence {float(conf):.2f} box {[round(float(v)) for v in b]}")
plt.figure(figsize=(7, 9)); plt.imshow(r.plot()[..., ::-1]); plt.axis("off")
plt.title("YOLO11n detections"); plt.show()
