import streamlit as st
import joblib
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, classification_report
from sklearn.decomposition import PCA
from scipy.sparse import hstack

# =========================
# LOAD FILES
# =========================
model = joblib.load("cybercrime_binary_model.pkl")
word_vec = joblib.load("word_vectorizer.pkl")
char_vec = joblib.load("char_vectorizer.pkl")
metrics = joblib.load("model_metrics.pkl")

# =========================
# SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

if "last_pred" not in st.session_state:
    st.session_state.last_pred = None

if "last_score" not in st.session_state:
    st.session_state.last_score = None

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# TITLE
# =========================
st.title("🚨 Cybercrime Detection System")

# =========================
# MODEL PERFORMANCE
# =========================
st.header("📊 Model Performance")

st.metric("Accuracy", round(metrics["accuracy"], 3))
st.metric("F1 Score", round(metrics["f1_score"], 3))

st.success("These scores are from unseen test data")

# =========================
# USER INPUT
# =========================
st.header("🧪 Test Message")

user_input = st.text_area("Enter message")

# ---------- PREDICT ----------
if st.button("Predict"):

    cleaned = clean_text(user_input)

    vec = hstack([
        word_vec.transform([cleaned]),
        char_vec.transform([cleaned])
    ])

    pred = model.predict(vec)[0]
    score = model.decision_function(vec)[0]

    st.session_state.last_pred = pred
    st.session_state.last_score = score

# ---------- SHOW RESULT ----------
if st.session_state.last_pred is not None:

    if st.session_state.last_pred == "scam":
        st.error("⚠️ SCAM MESSAGE")
        pred_num = 1
    else:
        st.success("✅ NORMAL MESSAGE")
        pred_num = 0

    true_label = st.selectbox("Actual label?", ["normal", "scam"])

    if st.button("Save Result"):
        st.session_state.history.append({
            "true": 1 if true_label == "scam" else 0,
            "pred": pred_num,
            "score": st.session_state.last_score
        })
        st.success("Saved! ✅")

# =========================
# USER-BASED VISUALIZATION
# =========================
if len(st.session_state.history) > 5:

    st.header("📊 User-Based Model Evaluation")

    hist_df = pd.DataFrame(st.session_state.history)

    y_true = hist_df["true"]
    y_pred = hist_df["pred"]
    scores = hist_df["score"]

    # Confusion Matrix
    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["Normal","Scam"],
                yticklabels=["Normal","Scam"],
                ax=ax)
    st.pyplot(fig)

    # ROC Curve
    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],"--")
    plt.title(f"AUC = {roc_auc:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    st.pyplot(fig)

    # Precision Recall
    st.subheader("📉 Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_true, scores)

    fig = plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    st.pyplot(fig)

    # Classification Report
    st.subheader("📄 Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

else:
    st.warning("⚠️ Add at least 6 user samples to see graphs")

# =========================
# 3D PCA VISUALIZATION (FIXED COLORS)
# =========================
st.header("🧊 SVM Data Distribution (Dataset Visualization)")

st.info("Red = Scam | Green = Normal")

data = pd.read_csv("phishing_email.csv")

sample_df = data.sample(500, random_state=42)

texts_small = sample_df["text_combined"].apply(clean_text)
labels_small = sample_df["label"]

vec_small = hstack([
    word_vec.transform(texts_small),
    char_vec.transform(texts_small)
]).toarray()

# PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(vec_small)

# 🔥 FIXED COLORS
colors = ["red" if label == 1 else "green" for label in labels_small]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    X_pca[:,0],
    X_pca[:,1],
    X_pca[:,2],
    c=colors,
    alpha=0.6
)

# Legend
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='Scam')
green_patch = mpatches.Patch(color='green', label='Normal')
ax.legend(handles=[red_patch, green_patch])

ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
ax.set_title("3D PCA Distribution")

st.pyplot(fig)