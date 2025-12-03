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
