import pandas as pd
import re
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle
from scipy.sparse import hstack

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("🚀 Loading datasets...")

datasets = []

# =========================
# CEAS
# =========================
ceas = pd.read_csv("CEAS_08.csv", encoding="latin-1")
ceas["text"] = ceas["subject"].astype(str) + " " + ceas["body"].astype(str)
ceas = ceas[["text","label"]]
datasets.append(ceas)

# =========================
# PHISHING
# =========================
phish = pd.read_csv("phishing_email.csv", encoding="latin-1")
phish = phish[["text_combined","label"]]
phish.columns = ["text","label"]
datasets.append(phish)

# =========================
# NIGERIAN FRAUD
# =========================
fraud = pd.read_csv("Nigerian_Fraud.csv", encoding="latin-1")
fraud["text"] = fraud["subject"].astype(str) + " " + fraud["body"].astype(str)
fraud = fraud[["text","label"]]
datasets.append(fraud)

# =========================
# ENRON
# =========================
enron = pd.read_csv("Enron.csv", encoding="latin-1")
enron["text"] = enron["subject"].astype(str) + " " + enron["body"].astype(str)
enron = enron[["text","label"]]
datasets.append(enron)

# =========================
# SPAM SMS
# =========================
spam = pd.read_csv("spam.csv", encoding="latin-1")
spam = spam.iloc[:,0:2]
spam.columns = ["label","text"]
spam["label"] = spam["label"].map({"ham":"normal","spam":"scam"})
datasets.append(spam)

# =========================
# SPAMASSASSIN
# =========================
spamass = pd.read_csv("SpamAssasin.csv", encoding="latin-1")

if "subject" in spamass.columns and "body" in spamass.columns:
    spamass["text"] = spamass["subject"].astype(str) + " " + spamass["body"].astype(str)

spamass = spamass[["text","label"]]

spamass["label"] = spamass["label"].astype(str).str.lower()
spamass["label"] = spamass["label"].replace({
    "ham":"normal",
    "spam":"scam",
    "0":"normal",
    "1":"scam"
})
datasets.append(spamass)

# =========================
# EMAILS.CSV (GENERIC)
# =========================
emails = pd.read_csv("emails.csv", encoding="latin-1")

# try to detect text column automatically
text_col = None
for col in emails.columns:
    if "text" in col.lower() or "body" in col.lower():
        text_col = col
        break

if text_col is not None:
    emails["text"] = emails[text_col]
    emails["label"] = "normal"   # assume normal emails
    emails = emails[["text","label"]]
    datasets.append(emails)

# =========================
# NORMAL DATASET FOLDER
# =========================
normal_folder = "normal-dataset"

if os.path.exists(normal_folder):
    normal_texts = []
    for file in os.listdir(normal_folder):
        try:
            with open(os.path.join(normal_folder, file), "r", encoding="latin-1") as f:
                normal_texts.append(f.read())
        except:
            continue

    normal_df = pd.DataFrame({
        "text": normal_texts,
        "label": ["normal"] * len(normal_texts)
    })
    datasets.append(normal_df)

# =========================
# COMBINE ALL
# =========================
df = pd.concat(datasets, ignore_index=True)

# =========================
# CLEANING
# =========================
df["text"] = df["text"].apply(clean_text)

df["label"] = df["label"].astype(str).str.lower()

df["label"] = df["label"].replace({
    "spam":"scam",
    "phishing":"scam",
    "fraud":"scam",
    "1":"scam",
    "ham":"normal",
    "0":"normal"
})

df = df[df["label"].isin(["normal","scam"])]
df = df.dropna()
df = df.drop_duplicates(subset=["text"])

df = shuffle(df, random_state=42)

print("\n📊 Final Dataset:")
print(df["label"].value_counts())

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# =========================
# VECTORIZERS
# =========================
word_vec = TfidfVectorizer(max_features=15000, ngram_range=(1,2), stop_words="english")
char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3,3), max_features=10000)

X_train_w = word_vec.fit_transform(X_train)
X_test_w = word_vec.transform(X_test)

X_train_c = char_vec.fit_transform(X_train)
X_test_c = char_vec.transform(X_test)

X_train_vec = hstack([X_train_w, X_train_c])
X_test_vec = hstack([X_test_w, X_test_c])

# =========================
# TRAIN SVM
# =========================
print("\n🚀 Training SVM...")

grid = GridSearchCV(
    LinearSVC(class_weight="balanced", max_iter=10000),
    {"C":[1,2,3]},
    scoring="f1_weighted",
    cv=3
)

grid.fit(X_train_vec, y_train)
model = grid.best_estimator_

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("\n✅ Accuracy:", acc)
print("✅ F1 Score:", f1)

# =========================
# SAVE
# =========================
joblib.dump(model, "cybercrime_binary_model.pkl")
joblib.dump(word_vec, "word_vectorizer.pkl")
joblib.dump(char_vec, "char_vectorizer.pkl")

joblib.dump({
    "accuracy": acc,
    "f1_score": f1
}, "model_metrics.pkl")

print("\n🔥 Model Saved Successfully!")