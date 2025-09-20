import os
import re
import io
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import requests
from bs4 import BeautifulSoup
import PyPDF2

# ML imports
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline



# For reproducibility
RANDOM_STATE = 42

app = Flask(__name__)
app.secret_key = "dev-secret-key"  # change for production
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"txt", "pdf"}

########################
# Demo training dataset
########################
# This is a small synthetic demo dataset. Replace with a large labeled dataset for production.
demo_texts = [
    # true news / factual statements
    ("The earth revolves around the sun.", "true"),
    ("Water freezes at 0 degrees Celsius at standard pressure.", "true"),
    ("Mount Everest is the highest mountain above sea level.", "true"),
    ("The chemical formula of water is H2O.", "true"),
    ("Smoking increases the risk of lung cancer.", "true"),
    # fake / false claims
    ("Vaccines implant microchips to control people.", "fake"),
    ("Drinking bleach cures COVID-19.", "fake"),
    ("The earth is flat and NASA hides it.", "fake"),
    ("5G towers spread viruses to humans.", "fake"),
    ("Eating only lemons makes you immune to diseases.", "fake"),
    # slightly ambiguous / mixed
    ("This celebrity supports policy X according to a screenshot I saw.", "fake"),
    ("A viral article claims product Y cures all ailments but lacks sources.", "fake"),
    ("Coffee is made from roasted beans.", "true"),
    ("Eating chocolate cures COVID-19.", "fake"),
    ("The sun rises in the east.", "true"),
    ("Aliens built the pyramids.", "fake")
]
texts = [t for t, label in demo_texts]
labels = [1 if label == "true" else 0 for t, label in demo_texts]  # 1=true, 0=fake

########################
# Train a simple model
########################
# TF-IDF -> LogisticRegression pipeline. Easy to explain (coef_ gives word importance).
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
clf = LogisticRegression(random_state=RANDOM_STATE, solver="liblinear")

pipeline = make_pipeline(vectorizer, clf)
pipeline.fit(texts, labels)

# grab vectorizer and classifier for explanation
tfidf: TfidfVectorizer = pipeline.named_steps['tfidfvectorizer'] if 'tfidfvectorizer' in pipeline.named_steps else pipeline.named_steps['tfidfvectorizer'] if False else pipeline.named_steps['tfidfvectorizer'] if False else pipeline.named_steps[list(pipeline.named_steps.keys())[0]]
# The above attempts to be robust across sklearn versions; simpler:
tfidf = pipeline.named_steps[list(pipeline.named_steps.keys())[0]]
classifier = pipeline.named_steps[list(pipeline.named_steps.keys())[1]]

# helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_stream):
    try:
        reader = PyPDF2.PdfReader(file_stream)
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)
    except Exception as e:
        return ""

def fetch_text_from_url(url):
    try:
        resp = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove scripts/styles
        for s in soup(["script", "style", "noscript"]):
            s.extract()
        texts = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        return "\n".join(texts)
    except Exception as e:
        return ""

def clean_text(t):
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t).strip()
    return t

def explain_prediction(text, top_n=8):
    """
    Returns top contributing tokens for true and fake by inspecting coef_ * tfidf features.
    """
    X_vec = tfidf.transform([text])
    # coef shape: (n_features,) for binary logistic regression with solver liblinear
    if hasattr(classifier, "coef_"):
        coefs = classifier.coef_[0]  # positive => pushes to label 1 (true)
    else:
        coefs = np.zeros(tfidf.vocabulary_._len_())
    feature_names = np.array(tfidf.get_feature_names_out())
    # compute contribution = coef * tfidf_value
    contributions = X_vec.toarray()[0] * coefs
    # positive contributions -> support true; negative -> support fake
    pos_idx = np.argsort(-contributions)[:top_n]
    neg_idx = np.argsort(contributions)[:top_n]
    pos = [(feature_names[i], float(contributions[i])) for i in pos_idx if contributions[i] > 0]
    neg = [(feature_names[i], float(contributions[i])) for i in neg_idx if contributions[i] < 0]
    return pos, neg

def interpret_probability(prob):
    # prob[1] is prob of "true"
    p_true = prob[0][1]
    if p_true >= 0.55:
        verdict = "True"
    elif p_true <= 0.45:
        verdict = "Fake"
    else:
        verdict = "Uncertain"
    return verdict, p_true

def generate_solutions(text, verdict):
    suggestions = []
    suggestions.append("Check reputable sources (major news outlets, academic papers, official statements).")
    suggestions.append("Find primary sources or official documents that support the claim.")
    suggestions.append("Reverse image search any images; check for manipulations.")
    suggestions.append("Look for fact-check articles from established fact-checkers (e.g., Snopes, FactCheck.org).")
    if verdict == "Fake":
        suggestions.insert(0, "Do not share the information until verified. It appears likely to be false.")
    elif verdict == "True":
        suggestions.insert(0, "Claim appears supported by available textual cues, but verify with original/primary sources.")
    else:
        suggestions.insert(0, "The model is uncertain. Manually verify using the steps below.")
    # tailor a bit
    if len(text) < 300:
        suggestions.append("If this is a short claim, try to find the sentence inside a longer article or a primary source for more context.")
    return suggestions

########################
# Flask routes
########################
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    input_text = ""
    # Priority: direct text > uploaded file > URL
    if request.form.get("input_type") == "text":
        input_text = request.form.get("text_input", "")
    elif request.form.get("input_type") == "file":
        if 'file_input' not in request.files:
            flash("No file part")
            return redirect(url_for('index'))
        file = request.files['file_input']
        if file.filename == '':
            flash("No file selected")
            return redirect(url_for('index'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            # read file
            ext = filename.rsplit('.', 1)[1].lower()
            if ext == "txt":
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    input_text = f.read()
            elif ext == "pdf":
                with open(path, "rb") as f:
                    input_text = extract_text_from_pdf(f)
            else:
                input_text = ""
        else:
            flash("Unsupported file type. Use .txt or .pdf.")
            return redirect(url_for('index'))
    elif request.form.get("input_type") == "url":
        url = request.form.get("url_input", "")
        input_text = fetch_text_from_url(url)
        if not input_text:
            flash("Could not fetch text from the URL or the page did not return usable textual paragraphs.")
            # still continue if the user provided short text in the text box
            return redirect(url_for('index'))
    else:
        # fallback to text
        input_text = request.form.get("text_input", "")

    input_text = clean_text(input_text)
    if not input_text:
        flash("No text to analyze. Provide text, a file, or a URL.")
        return redirect(url_for('index'))

    # predict
    prob = pipeline.predict_proba([input_text])  # returns [[prob_fake, prob_true]]
    verdict, p_true = interpret_probability(prob)
    pos, neg = explain_prediction(input_text, top_n=8)
    suggestions = generate_solutions(input_text, verdict)

    # minimal safety: trim long input for display
    short_preview = input_text[:2000] + ("..." if len(input_text) > 2000 else "")

    confidence = round(float(p_true) * 100, 2)
    return render_template("result.html",
                           verdict=verdict,
                           probability=confidence,
                           pos=pos,
                           neg=neg,
                           suggestions=suggestions,
                           preview=short_preview)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
#& "C:/Program Files/Python312/python.exe" -m pip install scikit-learn





