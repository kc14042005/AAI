import os, time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)
tf.random.set_seed(0); np.random.seed(0)

path = "tinyshakespeare.txt"
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists(path):
    path = keras.utils.get_file("tinyshakespeare.txt", url)
text = open(path, encoding="utf-8").read()
print("Corpus length:", len(text), "characters")
print(text[:300])

vocab = sorted(set(text))
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = np.array(vocab)
print("Unique characters:", len(vocab))
encoded = np.array([char2idx[c] for c in text], dtype=np.int32)

SEQ_LEN, BATCH = 100, 64
CORPUS = encoded[:600_000]
ds = tf.data.Dataset.from_tensor_slices(CORPUS).batch(SEQ_LEN + 1, drop_remainder=True)
ds = ds.map(lambda chunk: (chunk[:-1], chunk[1:]))
ds = ds.shuffle(10000, seed=0).batch(BATCH, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

for x, y in ds.take(1):
    print("Input :", repr("".join(idx2char[x[0][:40].numpy()])))
    print("Target:", repr("".join(idx2char[y[0][:40].numpy()])))

model = keras.Sequential([
    keras.Input(shape=(None,)),
    layers.Embedding(len(vocab), 64),
    layers.LSTM(384, return_sequences=True),
    layers.Dense(len(vocab))])
model.compile(optimizer=keras.optimizers.Adam(2e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.summary()

def generate(seed, length=400, temperature=0.8):
    ids = [char2idx[c] for c in seed]
    out = list(seed)
    for _ in range(length):
        logits = model(np.array([ids[-SEQ_LEN:]]), training=False)[0, -1].numpy() / temperature
        nxt = tf.random.categorical(logits[None, :], 1)[0, 0].numpy()
        ids.append(int(nxt)); out.append(idx2char[nxt])
    return "".join(out)

EPOCHS = 8
for epoch in range(1, EPOCHS + 1):
    t = time.time()
    h = model.fit(ds, epochs=1, verbose=0)
    print(f"Epoch {epoch}/{EPOCHS} - loss {h.history['loss'][0]:.4f} ({time.time() - t:.0f}s)")
    if epoch in (1, EPOCHS):
        print("Sample:", generate("ROMEO:", 200, 0.8).replace("\n", " / ")[:260], "\n")

for temp in [0.3, 0.8, 1.2]:
    print(f"===== temperature {temp} =====")
    print(generate("JULIET:\n", 350, temp), "\n")
model.save("char_lstm_shakespeare.keras")

import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, pipeline,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed)
from datasets import Dataset

device = 0 if torch.cuda.is_available() else -1
gpt_generator = pipeline("text-generation", model="gpt2", device=device)
set_seed(42)
for out in gpt_generator("The future of artificial intelligence in education", max_new_tokens=60,
                         num_return_sequences=2, do_sample=True, top_p=0.95):
    print(out["generated_text"], "\n---")

tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
blocks = [text[i:i + 2000] for i in range(0, 400_000, 2000)]
ds_hf = Dataset.from_dict({"text": blocks}).map(lambda b: tok(b["text"], truncation=True, max_length=256),
                                                batched=True, remove_columns=["text"])
args = TrainingArguments("gpt2-shakespeare", num_train_epochs=1, per_device_train_batch_size=8,
                         learning_rate=5e-5, logging_steps=20, save_strategy="no", report_to="none")
trainer = Trainer(model=gpt2, args=args, train_dataset=ds_hf,
                  data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
trainer.train()

ft = pipeline("text-generation", model=gpt2, tokenizer=tok, device=device)
print(ft("ROMEO:", max_new_tokens=120, do_sample=True, temperature=0.8, top_p=0.95)[0]["generated_text"])
