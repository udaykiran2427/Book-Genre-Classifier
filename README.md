# 📚 Book Genre Classifier

Predicts book genres from any text input using a machine learning model trained on Goodreads data.

🔗 **Live App:** https://book-genre-classifier.vercel.app/

---

## 🚀 Overview

This app identifies genres from a text passage using:

- **TF-IDF vectorization**
- **Linear SVM (One-vs-Rest)**
- **Multi-label classification**

Supported genres:

- Fantasy
- Romance
- Nonfiction
- Young Adult

Stack used:

- **Backend:** FastAPI (Render)
- **Frontend:** React + Vite (Vercel)
- **Model Files:** Stored with joblib

---

## 🧠 How It Works

1. User enters text into the UI
2. Frontend sends a `POST /predict` request
3. Backend vectorizes → runs classifier → returns genres
4. Frontend shows results
5. If input is too short or unclear, a fallback message is provided

---

## 🔌 API

### **POST /predict**

**Request**

```json
{ "text": "A kingdom rises as ancient magic awakens." }
```

**Response**

```json
{ "predicted_genres": ["Fantasy", "Young Adult"] }
```

**Fallback**

```json
{
  "predicted_genres": ["Could not determine genre. Provide more detailed text."]
}
```

---

## 🖥️ Run Locally

### Backend

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend →

```
http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend →

```
http://localhost:5173
```

---

Env:

```
VITE_API_URL=https://your-backend-url
```

---
