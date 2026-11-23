import os, json, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

print("Transformers version:", transformers.__version__)
model_name = "deepset/bert-base-cased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)

context = """
The Bengal tiger was chosen as the national animal in a meeting of the Indian wildlife board in 1972 and
was adopted officially in April 1973. It was chosen over the Asiatic lion due to the wider presence of the
tiger across India.
Indian peacock was designated as the national bird of India in February 1963.
Indian elephant is the largest terrestrial mammal in India and a cultural symbol throughout its range,
appearing in various religious traditions and mythologies.
Indian banyan is a large tree native to the Indian subcontinent and produces aerial roots from the
branches which grow downwards, eventually becoming trunks.
Mango is a large fruit tree with many varieties, believed to have originated in northeast India.
Lotus is an aquatic plant adapted to grow in the flood plains. Lotus seeds can remain dormant and viable
for many years, therefore the plant is regarded as a symbol of longevity.
"""

questions = ["What is the National animal of India?", "What is the National bird of India?",
             "What is the largest terrestrial mammal in India?", "Which tree produces aerial roots?",
             "Which fruit originated in northeast India?", "What is the symbol of longevity?"]

print("Encoded first question:", tokenizer.encode(questions[0]))

def chatbot(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = qa_model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    import torch
    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores) + 1

    all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    answer = tokenizer.convert_tokens_to_string(all_tokens[start_idx:end_idx])

    softmax = torch.nn.functional.softmax
    confidence = float(softmax(start_scores, dim=-1)[0][start_idx] * softmax(end_scores, dim=-1)[0][end_idx - 1])

    return answer, confidence

for q in questions:
    ans, score = chatbot(q, context)
    print(f"Q: {q}\nA: {ans} (confidence {score:.2f})\n")

tf.random.set_seed(7); np.random.seed(7); random.seed(7)

path = "intents.json"
if not os.path.exists(path):
    raise FileNotFoundError("Upload intents.json (provided with the journal) to the script folder")
with open(path) as f:
    intents = json.load(f)["intents"]

sentences, tags = [], []
for it in intents:
    for p in it["patterns"]:
        sentences.append(p); tags.append(it["tag"])
classes = sorted(set(tags))
tag2id = {t: i for i, t in enumerate(classes)}
y = np.array([tag2id[t] for t in tags])
print("Intents:", classes)
print("Training sentences:", len(sentences))

MAX_LEN = 12
vectorizer = layers.TextVectorization(max_tokens=1000, output_sequence_length=MAX_LEN)
vectorizer.adapt(sentences)
X = vectorizer(np.array(sentences)).numpy()
VOCAB = vectorizer.vocabulary_size()
print("Vocabulary size:", VOCAB)
print(sentences[0], "->", X[0])

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kw):
        super().__init__(**kw)
        self.token_emb = layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = layers.Embedding(maxlen, embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        return self.token_emb(x) + self.pos_emb(positions)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kw):
        super().__init__(**kw)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.norm1, self.norm2 = layers.LayerNormalization(epsilon=1e-6), layers.LayerNormalization(epsilon=1e-6)
        self.drop1, self.drop2 = layers.Dropout(rate), layers.Dropout(rate)

    def call(self, x, training=False):
        attn = self.att(x, x)
        x = self.norm1(x + self.drop1(attn, training=training))
        return self.norm2(x + self.drop2(self.ffn(x), training=training))

EMBED, HEADS, FF = 32, 2, 64
inputs = keras.Input(shape=(MAX_LEN,), dtype="int64")
x = TokenAndPositionEmbedding(MAX_LEN, VOCAB, EMBED)(inputs)
x = TransformerBlock(EMBED, HEADS, FF)(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(len(classes), activation="softmax")(x)
model = keras.Model(inputs, outputs, name="transformer_intent_classifier")
model.summary()

model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist = model.fit(X, y, epochs=150, batch_size=16, verbose=0)
print("Final training accuracy:", round(hist.history["accuracy"][-1], 4), "| loss:", round(hist.history["loss"][-1], 4))

test = [("hello there, anybody around?", "greeting"), ("how much do I have to pay for the course", "fees"),
        ("what is the procedure to join the college", "admission"), ("is data science offered here", "courses"),
        ("at what time does the college start", "timings"), ("how long can I keep a library book", "library"),
        ("when is the semester exam", "exam"), ("do companies come for campus recruitment", "placement"),
        ("give me your phone number", "contact"), ("thanks for helping me", "thanks"), ("ok bye", "goodbye"),
        ("where can I download the hall ticket", "exam")]

probs = model.predict(vectorizer(np.array([t for t, _ in test])), verbose=0)
correct = 0
for (q, true), p in zip(test, probs):
    pred = classes[p.argmax()]; correct += pred == true
    print(f"{q:<45} -> {pred:<10} ({p.max():.2f}) {'OK' if pred == true else 'expected ' + true}")
print(f"\nAccuracy on unseen questions: {correct}/{len(test)}")

responses = {it["tag"]: it["responses"] for it in intents}

def chat(message, threshold=0.45):
    p = model.predict(vectorizer(np.array([message])), verbose=0)[0]
    if p.max() < threshold:
        return "Sorry, I did not understand. Please ask about admissions, fees, courses, exams, library or placements."
    return random.choice(responses[classes[p.argmax()]])

conversation = ["Hi", "What courses do you offer?", "how much are the fees for MSc IT?",
                "when are the exams held?", "what is the weather in Paris?", "thank you", "bye"]
for m in conversation:
    print("You :", m)
    print("Chatbot:", chat(m), "\n")

while True:
    msg = input("You: ")
    if msg.lower() in ("quit", "exit"):
        break
    print("Chatbot:", chat(msg))
