# 📩 Gmail Email Classifier  
### AI-Powered Phishing & LLM Email Detection Chrome Extension

Gmail Email Classifier is a Chrome Extension that scans your Gmail inbox and classifies emails using a machine learning backend into:

- ✅ Legitimate (Human)
- 🤖 AI-Generated (LLM)
- ⚠️ Phishing (Human or LLM)

The extension injects visual badges directly into Gmail and provides a statistical summary with a pie chart inside the popup.

---

## 🚀 Features

- 🔍 One-click Gmail inbox scanning  
- 🎯 ML-based 4-class email classification  
- 🏷️ Visual badges added directly inside Gmail  
- 📊 Popup dashboard with:
  - Category counts
  - Pie chart visualization
  - Confidence scores
- ⚡ FastAPI backend deployed on Render  
- 🧠 TF-IDF + Logistic Regression model  

---

## 🏗️ Architecture

```
Chrome Extension (Content Script)
        ↓
Extract subject + snippet + sender
        ↓
POST request to Backend API
        ↓
FastAPI Server (Render)
        ↓
ML Model (TF-IDF + Logistic Regression)
        ↓
Prediction returned
        ↓
Badges applied in Gmail
        ↓
Stats shown in Popup Dashboard
```

---

## 🧠 Machine Learning Model

### Dataset

Trained on:

```
francescogreco97/human-llm-generated-phishing-legitimate-emails
```

### Classes

- `human_legit`
- `human_phishing`
- `llm_legit`
- `llm_phishing`

### Model Pipeline

```python
Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )),
    ("clf", LogisticRegression(
        max_iter=2500,
        class_weight="balanced"
    ))
])
```

### Input Format

Each email is converted into:

```
subject + snippet + sender
```

Example:

```
"Verify your account immediately Your password will expire support@bank.com"
```

---

## 🖥️ Backend (FastAPI)

### API Endpoint

```
POST /classify_emails
```

### Request Body

```json
{
  "emails": [
    {
      "id": "row-1",
      "from_": "support@example.com",
      "subject": "Verify your account",
      "snippet": "Your password will expire soon"
    }
  ]
}
```

### Response

```json
{
  "results": [
    {
      "id": "row-1",
      "label": "human_phishing",
      "score": 0.94
    }
  ]
}
```

---

## 📦 Installation (Developer Mode)

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/gmail-email-classifier.git
   ```

2. Open Chrome:
   ```
   chrome://extensions
   ```

3. Enable **Developer Mode**

4. Click **Load unpacked**

5. Select the extension folder

6. Open Gmail and click **Scan Inbox**

---

## 🌐 Deployment (Backend)

Backend is deployed on:

```
Render
```

To deploy manually:

1. Push backend code to GitHub  
2. Create new Web Service on Render  
3. Set:
   - Build command:
     ```
     pip install -r requirements.txt
     ```
   - Start command:
     ```
     uvicorn main:app --host 0.0.0.0 --port 10000
     ```

4. Add environment variable:
   ```
   EMAIL_MODEL_PATH=email_classifier_4class.joblib
   ```

---

## 🔐 Privacy & Data Handling

- Only visible email subject, snippet, and sender are read.
- Data is sent to backend only when user clicks **Scan Inbox**.
- No permanent storage of email content.
- Users can uninstall the extension at any time.

---

## ⚠️ Known Limitations

- Model may misclassify promotional emails as phishing.
- TF-IDF model does not understand context deeply.
- Gmail DOM changes may require selector updates.
- Real-world accuracy depends on dataset diversity.

---

## 🔮 Future Improvements

- Replace TF-IDF with Transformer model (DistilBERT)
- Add Gmail API integration
- Add real-time background scanning
- Add user feedback loop for model improvement
- Improve false-positive reduction

---

## 👨‍💻 Author

Built for hackathon / cybersecurity experimentation.

---

## 📜 License

MIT License

---

## ⭐ If you like this project

Star the repo and contribute!
