'''
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax for hackathon
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Email(BaseModel):
    id: str
    from_: str | None = None
    subject: str
    snippet: str

class EmailRequest(BaseModel):
    emails: List[Email]

class EmailResult(BaseModel):
    id: str
    label: str
    score: float

class EmailResponse(BaseModel):
    results: List[EmailResult]

# 🔹 Load ML model at startup
print("Loading email classifier model...")
model = joblib.load("email_classifier.joblib")
print("Model loaded.")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/classify_emails", response_model=EmailResponse)
def classify_emails(req: EmailRequest):
    texts = []
    ids = []

    for email in req.emails:
        text = f"{email.subject} {email.snippet} {email.from_ or ''}"
        texts.append(text)
        ids.append(email.id)

    # 🔹 Use the ML model to predict
    preds = model.predict(texts)             # labels: phishing / ai_generated / legit
    probs = model.predict_proba(texts)       # probability scores
    classes = list(model.classes_)           # class order for probs

    results: List[EmailResult] = []

    for i, email_id in enumerate(ids):
        label = preds[i]
        # max probability for chosen label
        score = float(max(probs[i]))

        results.append(EmailResult(id=email_id, label=label, score=score))

    return EmailResponse(results=results)
'''
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax for hackathon
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic models ----------

class Email(BaseModel):
    id: str
    from_: str | None = None
    subject: str
    snippet: str

class EmailRequest(BaseModel):
    emails: List[Email]

class EmailResult(BaseModel):
    id: str
    label: str   # one of: human_legit / human_phishing / llm_legit / llm_phishing
    score: float # model confidence for chosen label

class EmailResponse(BaseModel):
    results: List[EmailResult]


# ---------- Load ML model at startup ----------

MODEL_PATH = os.getenv("EMAIL_MODEL_PATH", "email_classifier_4class.joblib")

print(f"Loading email classifier model from {MODEL_PATH} ...")
model = joblib.load(MODEL_PATH)
print("Model loaded.")
print("Model classes:", model.classes_)  # useful debug


@app.get("/")
def root():
    return {"status": "ok", "model": "email_classifier_4class"}


@app.post("/classify_emails", response_model=EmailResponse)
def classify_emails(req: EmailRequest):
    texts: List[str] = []
    ids: List[str] = []

    # Build text input for the model
    for email in req.emails:
        # simple concat of subject + snippet + from
        text = f"{email.subject} {email.snippet} {email.from_ or ''}"
        texts.append(text)
        ids.append(email.id)

    # 🔹 Predict with 4-class model
    preds = model.predict(texts)        # labels: human_legit, human_phishing, llm_legit, llm_phishing
    probs = model.predict_proba(texts)  # probability scores

    results: List[EmailResult] = []

    for i, email_id in enumerate(ids):
        label = str(preds[i])
        score = float(max(probs[i]))  # highest probability among the 4 classes

        results.append(EmailResult(
            id=email_id,
            label=label,
            score=score
        ))

    return EmailResponse(results=results)
