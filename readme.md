# 🕵️‍♂️ Dark Web & Cybercrime Monitoring

A machine learning-powered solution designed to detect and flag suspicious cybercrime activities. By leveraging **Natural Language Processing (NLP)**, this tool analyzes text inputs to identify patterns associated with phishing, fraud, and illicit communication.

---

## 🚀 Key Features
* **Real-time Detection:** Immediate classification of suspicious text via a web interface.
* **NLP Pipeline:** Implementation of **TF-IDF** (Term Frequency-Inverse Document Frequency) for high-accuracy text vectorization.
* **Multi-Dataset Training:** The model is trained on diverse datasets, including the Enron corpus and various phishing samples, to ensure robustness.

## 🛠️ The Tech Stack
* **Language:** Python 3.x
* **ML Framework:** Scikit-learn (Logistic Regression/Naive Bayes)
* **Data Handling:** Pandas & NumPy
* **Deployment:** Streamlit (Web UI)
* **Model Persistence:** Joblib/Pickle

## 📂 Project Structure
* `train.py`: The "Engine" — handles data cleaning, TF-IDF transformation, and model training.
* `app.py`: The "Face" — a Streamlit application that provides the user interface.
* `*.pkl`: Pre-trained weights and vectorizers for instant predictions without re-training.

## ⚙️ Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/dhanushr222/NLP---Cybercrime-detection.git](https://github.com/dhanushr222/NLP---Cybercrime-detection.git)