import os, sys, time, subprocess, textwrap
import joblib, requests, sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    print("scikit-learn:", sklearn.__version__)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2,
                                                        stratify=iris.target, random_state=1)
    model = Pipeline([("scale", StandardScaler()),
                      ("clf", RandomForestClassifier(n_estimators=200, random_state=1))])
    model.fit(X_train, y_train)
    print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")

    os.makedirs("iris_app", exist_ok=True)
    joblib.dump({"model": model, "classes": list(iris.target_names), "version": "1.0.0"},
                "iris_app/model.joblib")
    print("Saved iris_app/model.joblib,", os.path.getsize("iris_app/model.joblib"), "bytes")

    app_code = '''
from contextlib import asynccontextmanager
from typing import List
import joblib, numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

bundle = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle.update(joblib.load("model.joblib"))
    yield

app = FastAPI(title="Iris Species Classifier", version="1.0.0", lifespan=lifespan)

class Flower(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])

class Batch(BaseModel):
    flowers: List[Flower]

@app.get("/health")
def health():
    return {"status": "ok", "model_version": bundle.get("version")}

def _predict(flowers):
    X = np.array([[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width] for f in flowers])
    proba = bundle["model"].predict_proba(X)
    return [{"species": bundle["classes"][int(p.argmax())], "confidence": round(float(p.max()), 3)} for p in proba]

@app.post("/predict")
def predict(flower: Flower):
    return _predict([flower])[0]

@app.post("/predict_batch")
def predict_batch(batch: Batch):
    if not batch.flowers:
        raise HTTPException(status_code=400, detail="empty batch")
    return {"predictions": _predict(batch.flowers)}
'''
    with open("iris_app/main.py", "w") as f:
        f.write(textwrap.dedent(app_code).lstrip())
    print(open("iris_app/main.py").read()[:400], "...")

    requirements = f"""fastapi>=0.110
uvicorn[standard]>=0.29
scikit-learn=={sklearn.__version__}
joblib>=1.3
numpy>=1.26
"""
    dockerfile = """FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py model.joblib ./
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
"""
    dockerignore = "__pycache__/\n*.ipynb\n.ipynb_checkpoints/\n"
    for name, content in [("requirements.txt", requirements), ("Dockerfile", dockerfile),
                          (".dockerignore", dockerignore)]:
        with open(f"iris_app/{name}", "w") as f:
            f.write(content)
    print(open("iris_app/requirements.txt").read()); print(open("iris_app/Dockerfile").read())
    print("Files in iris_app:", sorted(os.listdir("iris_app")))

    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--port", "8000"], cwd="iris_app",
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        try:
            if requests.get("http://127.0.0.1:8000/health", timeout=1).ok: break
        except requests.ConnectionError:
            time.sleep(0.5)

    BASE = "http://127.0.0.1:8000"
    print("GET /health ->", requests.get(f"{BASE}/health").json())
    sample = {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}
    print("POST /predict", sample, "->", requests.post(f"{BASE}/predict", json=sample).json())

    batch = {"flowers": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                         {"sepal_length": 5.9, "sepal_width": 2.8, "petal_length": 4.4, "petal_width": 1.3},
                         {"sepal_length": 7.7, "sepal_width": 2.6, "petal_length": 6.9, "petal_width": 2.3}]}
    print("POST /predict_batch ->")
    for p in requests.post(f"{BASE}/predict_batch", json=batch).json()["predictions"]:
        print(" ", p)

    bad = requests.post(f"{BASE}/predict", json={"sepal_length": -1, "sepal_width": 3})
    print("Invalid input -> HTTP", bad.status_code, "|", bad.json()["detail"][0]["msg"])
    print("Interactive API docs (Swagger UI) available at", f"{BASE}/docs")

    t = time.time()
    for _ in range(200):
        requests.post(f"{BASE}/predict", json=sample)
    print(f"Average latency over 200 requests: {(time.time() - t) / 200 * 1000:.1f} ms")

    server.terminate(); server.wait(); print("Local server stopped")
