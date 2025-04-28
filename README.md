# Sentiment Analysis Model 😊☹️

This repository contains a transformer-based sentiment classification pipeline, combining weak supervision with Snorkel and rule-based labeling, followed by fine-tuning a DistilBERT encoder. It also includes a simple HTML/CSS/JavaScript frontend and a Flask backend for deployment, enabling end-to-end sentiment prediction via a web interface.

---

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Model Architecture](#model-architecture)
4. [Data Preprocessing](#data-preprocessing)
5. [Weak Supervision Labeling](#weak-supervision-labeling)
6. [Fine-Tuning & Training](#fine-tuning--training)
7. [Evaluation](#evaluation)
8. [Saving & Loading the Model](#saving--loading-the-model)
9. [Web Application (Frontend & Backend)](#web-application-frontend--backend)
10. [Usage Example](#usage-example)
11. [License](#license)

---

## Features

- **Weak supervision** via Snorkel labeling functions (regex- and rule-based heuristics).
- **Pretrained transformer** (DistilBERT) backbone for language understanding.
- **Customizable training pipeline** using Hugging Face’s `Trainer`.
- **Rule-based sentiment scoring** fallback with VADER.
- **Simple web UI** built with HTML, CSS, and JavaScript.
- **Flask API** for serving predictions.

---

## Requirements

List of key dependencies (also listed in `requirements.txt`):

- `torch` (PyTorch)
- `transformers` (Hugging Face)
- `snorkel`
- `spacy`
- `vaderSentiment`
- `scikit-learn`
- `datasets` (Hugging Face)
- `pandas`, `numpy`, `matplotlib`
- `Flask`, `Flask-CORS`
- `pdfplumber`
- `pytesseract`, `Pillow`

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Model Architecture

At its core, the pipeline fine-tunes **DistilBERT** (`distilbert-base-uncased`) for binary classification:

1. **Tokenizer**: Hugging Face `AutoTokenizer.from_pretrained("distilbert-base-uncased")`
2. **Encoder**: `AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)`
3. **Data collator**: `DataCollatorWithPadding` for dynamic padding during batching.
4. **Trainer**: `Trainer` API handles optimization, evaluation, and checkpointing.

---

## Data Preprocessing

1. **Load raw data** (CSV or JSON) into a pandas DataFrame.
2. **Text cleaning**: regex-based normalization, lowercasing, removal of non-ASCII characters.
3. **spaCy tokenization** (optional) for more advanced linguistic features.
4. **Train/test split** via `sklearn.model_selection.train_test_split`.
5. **Convert to Hugging Face Dataset** for seamless integration with the `Trainer`.

---

## Weak Supervision Labeling

To bootstrap label quality before transformer training:

- Define multiple **Snorkel labeling functions** (LFs) leveraging:

  - Regular expressions for positive/negative keywords.
  - VADER sentiment scores as a heuristic LF.
  - Part-of-speech patterns via spaCy.

- Apply LFs with `PandasLFApplier` and analyze coverage/conflict with `LFAnalysis`.

- Generate a probabilistic label matrix and threshold to produce silver-standard training labels.

---

## Fine-Tuning & Training

Training is orchestrated with Hugging Face’s `Trainer`:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="outputs/",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir="logs/",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()
```

Key hyperparameters (editable in the notebook):

- **Learning rate**: e.g. 5e-5
- **Batch size**: 8–32
- **Epochs**: 2–5
- **Warmup steps**, weight decay, etc.

---

## Evaluation

After training, evaluation metrics are computed on the held-out test set. Key metrics include:

- **Accuracy**: 0.9276 = 92.76%

- **Precision / Recall / F1**:

  | Class            | Precision | Recall | F1-score | Support |
  | ---------------- | --------- | ------ | -------- | ------- |
  | NEGATIVE         | 0.93      | 0.93   | 0.93     | 465     |
  | POSITIVE         | 0.92      | 0.93   | 0.92     | 419     |
  | **Macro avg**    | 0.93      | 0.93   | 0.93     | 884     |
  | **Weighted avg** | 0.93      | 0.93   | 0.93     | 884     |

- **Confusion matrix**:

  ```text
  [[431  34]
  [ 30 389]]
  ```

This comprehensive evaluation ensures you understand both overall performance (accuracy) and class-specific behavior (precision, recall, F1).

\--- (accuracy) and class-specific behavior (precision, recall, F1).

---

## Saving & Loading the Model

Once training completes:

```python
model.save_pretrained("sentiment_model")
tokenizer.save_pretrained("sentiment_model")
```

You can reload for inference:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sentiment_model")
model     = AutoModelForSequenceClassification.from_pretrained("sentiment_model")
```

---

## Web Application (Frontend & Backend)

The repository includes a web interface with the following components. The code is already provided; here are instructions to deploy and run:

- **Frontend**:

  - Check:
    - `index.html` — main HTML page with input form and file upload.
    - `styles.css` — styling for the web interface.
    - &#x20;`script.js` — JavaScript handling form submission, AJAX requests to the Flask API, and DOM updates.
  - Provides a form to submit either text input or upload a PDF/image file for sentiment analysis.

- **Backend** (Flask)\*\*:

  - Entry point: `app.py` (or equivalent).
  - Dependencies: Flask, Flask-CORS, Hugging Face Transformers, pdfplumber, pytesseract, and PIL.
  - Endpoints:
    - `GET /` : Serves the HTML frontend.
    - `POST /predict` : Accepts JSON payload `{ "text": "..." }`, returns `{ "sentiment": "positive|negative", "confidence": float }`.
    - `POST /upload` : Accepts multipart file uploads (`file` field). Supports `.pdf`, `.png`, `.jpg`, and `.jpeg`.
      - PDF: extracts text via pdfplumber.
      - Image: extracts text via Tesseract OCR.
      - Returns the same JSON schema as `/predict`.

- **Running the App**:

  1. Ensure the trained model directory (`sentiment_model/`) is in the project root.
  2. Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
  3. Set environment variable and start Flask:
     ```bash
     export FLASK_APP=app.py
     flask run
     ```
  4. Open `http://localhost:5000` (or whichever port you are running)in your browser to use the application.

---

## Usage Example

```bash
# Start the Flask app
flask run
```

Visit `http://localhost:5000`, enter text, and view the predicted label and confidence score.

---

## License

MIT License

Copyright (c) 2025 omprakash rout

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Contact 

**Email** - omprakashrouttt1@gmail.com

*Last updated: April 28, 2025*

