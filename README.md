
### 👥 Team: Combat Info Verifiers  

--

## 📌 Problem Statement

The spread of misinformation through social media, news platforms, and documents leads to **public confusion, panic, and distrust**.  
There is a need for an **AI-powered tool** that can quickly verify claims and provide explainable insights.

---

## 🚀 Prototype Overview

This is a **Flask-based web application** that analyzes input content (Text, URLs, or Files) to detect whether the content is:

- ✅ True  
- ⚠️ Misleading  
- ❌ Fake  

It also provides **confidence scores** and **explanation** using keyword highlighting.

---

## ✨ Features

- ✅ Detects **Fake / True / Misleading** content
- 📊 Displays **confidence score**
- 🟡 Highlights **suspicious or strong keywords**
- 🧾 Supports **Text, URL, PDF, and TXT** inputs
- ⚙️ Simple **Bootstrap UI**
- 🗂️ Tabs, Sample Inputs, and Suggestions

---

## 🔍 Process Flow

1. User submits Text / URL / File  
2. Text is extracted and cleaned  
3. ML pipeline: `TF-IDF → Logistic Regression`  
4. Prediction made with confidence  
5. Keyword explainability + verification tips  
6. Output shown on web dashboard

---

## 🧱 Architecture

- **Frontend:** HTML + Bootstrap  
- **Backend:** Flask (Python)  
- **ML Model:** TF-IDF + Logistic Regression  
- **Libraries:** Scikit-learn, BeautifulSoup, PyPDF2, Requests  
- **Deployment:** Flask server (locally; scalable to Docker/Cloud)

---

## 🛠️ Tech Stack

| Category     | Tools / Libraries                  |
|--------------|------------------------------------|
| Web          | Flask, HTML, Bootstrap             |
| ML Pipeline  | Scikit-learn, NumPy                |
| Text Tools   | BeautifulSoup, Requests, PyPDF2    |

---

## 💰 Cost Estimate

- 💻 **Prototype:** Local Flask app – no cost  
- ☁️ **Scale-up:** Hosting, Dataset licensing (~\$50–100/month)

---

## 📎 Note

This is a **prototype** trained on a small dataset for demonstration only.  
Model accuracy and generalizability can be improved with:

- Real-world data (e.g. from Kaggle, Fact-check sites)  
- Better ML models (e.g. BERT, LLMs)  
- Multilingual support & API integration

---


## 🤝 Contributions

Feel free to fork and extend this project!  
PRs are welcome.

---
