import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score

print("scikit-learn:", sklearn.__version__)

path = "titanic.csv"
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(path if os.path.exists(path) else url)
print(df.shape); print(df.head())
print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])
print("\nSurvival rate:", round(df.Survived.mean(), 3))

X = df.drop(columns=["Survived", "PassengerId"])
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def engineer_features(d):
    d = d.copy()
    d["Title"] = d["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    d["Title"] = d["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    d.loc[~d["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Title"] = "Rare"
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"] = (d["FamilySize"] == 1).astype(int)
    d["FarePerPerson"] = d["Fare"] / d["FamilySize"]
    d["HasCabin"] = d["Cabin"].notna().astype(int)
    d["Deck"] = d["Cabin"].str[0].fillna("U")
    d["TicketPrefix"] = d["Ticket"].str.contains(r"[A-Za-z]").astype(int)
    d["AgeMissing"] = d["Age"].isna().astype(int)
    return d.drop(columns=["Name", "Ticket", "Cabin"])

fe = FunctionTransformer(engineer_features)
print(fe.fit_transform(X_train).head())

numeric = make_column_selector(dtype_include=np.number)
categorical = make_column_selector(dtype_exclude=np.number)
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)])

pipe = Pipeline([("features", fe), ("preprocess", preprocess),
                 ("select", SelectKBest(mutual_info_classif, k=15)),
                 ("model", LogisticRegression(max_iter=1000))])

param_grid = [
    {"model": [LogisticRegression(max_iter=2000)], "model__C": [0.1, 1, 10], "select__k": [10, 15, "all"]},
    {"model": [RandomForestClassifier(random_state=42)], "model__n_estimators": [200],
     "model__max_depth": [4, 6, None], "select__k": [15, "all"]},
    {"model": [HistGradientBoostingClassifier(random_state=42)], "model__learning_rate": [0.05, 0.1],
     "model__max_depth": [3, None], "select__k": [15, "all"]},
    {"model": [SVC(probability=True, random_state=42)], "model__C": [0.5, 1, 5], "model__gamma": ["scale"],
     "select__k": [15, "all"]},
    {"model": [KNeighborsClassifier()], "model__n_neighbors": [5, 9, 15], "select__k": [10, 15]},
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
search.fit(X_train, y_train)
print("Pipelines evaluated:", len(search.cv_results_["params"]), "x 5 folds")
print("Best CV accuracy :", round(search.best_score_, 4))
print("Best pipeline    :", {k: (type(v).__name__ if k == "model" else v) for k, v in search.best_params_.items()})

res = pd.DataFrame(search.cv_results_)
res["algorithm"] = res["param_model"].apply(lambda m: type(m).__name__)
best_per_algo = res.sort_values("mean_test_score", ascending=False).groupby("algorithm").head(1)
print(best_per_algo[["algorithm", "mean_test_score", "std_test_score"]].round(4).to_string(index=False))

plt.figure(figsize=(7, 3))
plt.barh(best_per_algo["algorithm"], best_per_algo["mean_test_score"],
         xerr=best_per_algo["std_test_score"], color="tab:blue")
plt.xlim(0.7, 0.9); plt.xlabel("5-fold CV accuracy"); plt.title("Best configuration of each algorithm"); plt.show()

best = search.best_estimator_
y_pred = best.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, target_names=["Died", "Survived"]))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Died", "Survived"], cmap="Blues")
plt.show()

names = best.named_steps["preprocess"].get_feature_names_out()
selected = names[best.named_steps["select"].get_support()]
print("Features kept by SelectKBest:", [s.split("__")[1] for s in selected])

joblib.dump(best, "titanic_best_pipeline.joblib")
new_passenger = pd.DataFrame([{"Pclass": 1, "Name": "Doe, Mrs. Jane", "Sex": "female", "Age": 30,
                               "SibSp": 1, "Parch": 0, "Ticket": "PC 17599", "Fare": 80.0,
                               "Cabin": "C85", "Embarked": "C"}])
print("Saved pipeline; survival probability for a new passenger:",
      round(joblib.load("titanic_best_pipeline.joblib").predict_proba(new_passenger)[0, 1], 3))
