"""
Fake News Detection using Machine Learning
Pipeline: Text cleaning -> TF-IDF features -> Classifiers -> Evaluation

Usage:
  python fake_news_detection.py                      # uses built-in demo data
  python fake_news_detection.py True.csv Fake.csv    # uses the Kaggle
                                                     # "Fake and Real News" dataset
"""
import re
import sys
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

random.seed(42)
np.random.seed(42)

# ----------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------
def make_demo_data(n=3000):
    """Synthetic demo corpus (used only when no real CSV files are given).
    Real and fake articles share a lot of neutral vocabulary, and 4% of
    labels are flipped, so the task is not trivially separable."""
    topics = ["the government", "the health ministry", "scientists", "the central bank",
              "the election commission", "a local school", "the railway department",
              "researchers", "the police", "a tech company"]
    neutral = ["announced", "reported", "said", "confirmed", "released", "published",
               "new figures", "a statement", "this week", "on Monday", "officials",
               "the report", "according to", "data shows", "policy", "city", "state"]
    real_words = ["study", "evidence", "committee", "approved", "audit", "survey",
                  "official", "press conference", "peer-reviewed", "statistics",
                  "spokesperson", "budget", "investigation", "documents", "verified"]
    fake_words = ["shocking", "secret", "exposed", "hoax", "unbelievable", "banned",
                  "conspiracy", "miracle", "cover-up", "they don't want you to know",
                  "bombshell", "leaked", "destroyed", "outrage", "must share", "100% proof"]
    rows = []
    for _ in range(n):
        label = random.choice([0, 1])               # 1 = FAKE, 0 = REAL
        # 25% of articles use vocabulary of the opposite class (noise/overlap)
        cue = fake_words if (label == 1) ^ (random.random() < 0.25) else real_words
        words = [random.choice(topics)]
        for _ in range(random.randint(14, 30)):
            words.append(random.choice(cue) if random.random() < 0.25
                         else random.choice(neutral + topics))
        text = " ".join(words)
        if random.random() < 0.04:                   # label noise
            label = 1 - label
        rows.append((text, label))
    return pd.DataFrame(rows, columns=["text", "label"])


def load_data(argv):
    if len(argv) >= 3:
        real = pd.read_csv(argv[1]); fake = pd.read_csv(argv[2])
        real["label"], fake["label"] = 0, 1
        df = pd.concat([real, fake], ignore_index=True)
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
        print(f"Loaded real dataset: {len(df)} articles")
        return df[["text", "label"]].sample(frac=1, random_state=42)
    print("No CSV supplied -> using built-in synthetic demo data")
    return make_demo_data()

# ----------------------------------------------------------------------
# 2. PRE-PROCESSING
# ----------------------------------------------------------------------
def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)      # URLs
    text = re.sub(r"[^a-z\s']", " ", text)           # punctuation/digits
    return re.sub(r"\s+", " ", text).strip()

df = load_data(sys.argv)
df["text"] = df["text"].apply(clean)
print(f"Articles: {len(df)} | REAL: {(df.label==0).sum()} | FAKE: {(df.label==1).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

# ----------------------------------------------------------------------
# 3. FEATURE EXTRACTION (TF-IDF, unigrams + bigrams)
# ----------------------------------------------------------------------
tfidf = TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1, 2),
                        max_features=20000)
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)
print(f"TF-IDF matrix: train {Xtr.shape}, test {Xte.shape}")

# ----------------------------------------------------------------------
# 4. MODELS
# ----------------------------------------------------------------------
models = {
    "Logistic Regression":       LogisticRegression(max_iter=1000),
    "Multinomial Naive Bayes":   MultinomialNB(),
    "Passive Aggressive":        PassiveAggressiveClassifier(max_iter=50, random_state=42),
    "Random Forest":             RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}

results, preds = [], {}
for name, model in models.items():
    model.fit(Xtr, y_train)
    p = model.predict(Xte)
    preds[name] = p
    results.append({
        "Model": name,
        "Accuracy":  accuracy_score(y_test, p),
        "Precision": precision_score(y_test, p),
        "Recall":    recall_score(y_test, p),
        "F1-score":  f1_score(y_test, p),
    })

res = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
print("\n=== MODEL COMPARISON (test set) ===")
print(res.round(4).to_string(index=False))

best = res.loc[0, "Model"]
print(f"\nBest model: {best}")
print("\n=== CLASSIFICATION REPORT:", best, "===")
print(classification_report(y_test, preds[best], target_names=["REAL", "FAKE"], digits=4))

cm = confusion_matrix(y_test, preds[best])
print("Confusion matrix [rows=actual, cols=predicted] (REAL, FAKE):")
print(cm)

# ----------------------------------------------------------------------
# 5. PLOTS
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ConfusionMatrixDisplay(cm, display_labels=["REAL", "FAKE"]).plot(ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title(f"Confusion Matrix - {best}")
ax[1].barh(res["Model"][::-1], res["Accuracy"][::-1], color="#4c72b0")
ax[1].set_xlim(0.5, 1.0); ax[1].set_xlabel("Accuracy"); ax[1].set_title("Model accuracy")
for i, v in enumerate(res["Accuracy"][::-1]):
    ax[1].text(v + 0.005, i, f"{v:.3f}", va="center")
plt.tight_layout(); plt.savefig("results.png", dpi=150)

# ----------------------------------------------------------------------
# 6. TOP INDICATIVE WORDS (Logistic Regression)
# ----------------------------------------------------------------------
lr = models["Logistic Regression"]
names = np.array(tfidf.get_feature_names_out())
order = np.argsort(lr.coef_[0])
print("\nTop words pointing to REAL:", list(names[order[:8]]))
print("Top words pointing to FAKE:", list(names[order[-8:]][::-1]))

# ----------------------------------------------------------------------
# 7. PREDICT NEW TEXT
# ----------------------------------------------------------------------
best_model = models[best]
samples = [
    "Shocking secret exposed: leaked bombshell proves the cover-up, must share before it is banned",
    "The health ministry released new figures this week, and the committee approved the audited budget",
]
print("\n=== PREDICTIONS ON NEW TEXT ===")
for s in samples:
    label = best_model.predict(tfidf.transform([clean(s)]))[0]
    print(f"[{'FAKE' if label else 'REAL'}] {s}")
