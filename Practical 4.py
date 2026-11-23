import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))
tf.random.set_seed(1)

(train_images, train_labels), _ = keras.datasets.fashion_mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1).astype("float32")
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE, BATCH_SIZE, NOISE_DIM = 60000, 128, 100
dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print("Training images:", train_images.shape)

fig, ax = plt.subplots(1, 10, figsize=(13, 1.6))
for i in range(10):
    ax[i].imshow(train_images[i, :, :, 0], cmap="gray"); ax[i].axis("off")
plt.suptitle("Real Fashion-MNIST images"); plt.show()

def make_generator():
    return keras.Sequential([
        keras.Input(shape=(NOISE_DIM,)),
        layers.Dense(7 * 7 * 256, use_bias=False), layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, 5, strides=1, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(64, 5, strides=2, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(1, 5, strides=2, padding="same", activation="tanh")], name="generator")

def make_discriminator():
    return keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, 5, strides=2, padding="same"), layers.LeakyReLU(0.2), layers.Dropout(0.3),
        layers.Conv2D(128, 5, strides=2, padding="same"), layers.LeakyReLU(0.2), layers.Dropout(0.3),
        layers.Flatten(), layers.Dense(1)], name="discriminator")

generator, discriminator = make_generator(), make_discriminator()
generator.summary(); discriminator.summary()

cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    generator_optimizer.apply_gradients(zip(gen_tape.gradient(gen_loss, generator.trainable_variables),
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_tape.gradient(disc_loss, discriminator.trainable_variables),
                                                discriminator.trainable_variables))
    return gen_loss, disc_loss

seed = tf.random.normal([25, NOISE_DIM])

def show_samples(epoch):
    pred = generator(seed, training=False)
    fig = plt.figure(figsize=(5, 5))
    for i in range(25):
        plt.subplot(5, 5, i + 1); plt.imshow(pred[i, :, :, 0] * 127.5 + 127.5, cmap="gray"); plt.axis("off")
    plt.suptitle(f"Generated images after epoch {epoch}"); plt.show()

EPOCHS = 15
g_hist, d_hist = [], []
for epoch in range(1, EPOCHS + 1):
    start = time.time(); gl, dl = [], []
    for image_batch in dataset:
        g, d = train_step(image_batch); gl.append(float(g)); dl.append(float(d))
    g_hist.append(np.mean(gl)); d_hist.append(np.mean(dl))
    print(f"Epoch {epoch:2d}/{EPOCHS} generator loss {g_hist[-1]:.4f} "
          f"discriminator loss {d_hist[-1]:.4f} ({time.time() - start:.0f}s)")
    if epoch in (1, 5, EPOCHS):
        show_samples(epoch)

plt.figure(figsize=(7, 3.5))
plt.plot(range(1, EPOCHS + 1), g_hist, "o-", label="Generator")
plt.plot(range(1, EPOCHS + 1), d_hist, "o-", label="Discriminator")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("GAN training losses"); plt.legend(); plt.show()

new = generator(tf.random.normal([8, NOISE_DIM]), training=False)
d_fake = tf.sigmoid(discriminator(new, training=False)).numpy().ravel()
d_real = tf.sigmoid(discriminator(train_images[:8], training=False)).numpy().ravel()
print("Discriminator P(real) for 8 real images     :", d_real.round(2))
print("Discriminator P(real) for 8 generated images:", d_fake.round(2))

fig, ax = plt.subplots(1, 8, figsize=(12, 1.8))
for i in range(8):
    ax[i].imshow(new[i, :, :, 0], cmap="gray"); ax[i].set_title(f"{d_fake[i]:.2f}", fontsize=8); ax[i].axis("off")
plt.suptitle("Brand-new generated images (title = discriminator's P(real))"); plt.show()

generator.save("fashion_dcgan_generator.keras")
print("Generator saved to fashion_dcgan_generator.keras")
