# Brain MRI Tumor Detection Project Documentation

This is a complete deep learning system that classifies brain MRI scans into **Glioma, Meningioma, Pituitary Tumor, or No Tumor** using transfer learning (EfficientNetB0). It generates an AI-assisted explanatory report using the Gemini LLM and presents everything through a Streamlit web application with SQLite for prediction history.

---

## 1. Dataset

| Detail | Value |
|---|---|
| **Name** | Brain Tumor MRI Dataset |
| **Classes** | Glioma, Meningioma, Pituitary Tumor, No Tumor |
| **Total images** | 7,000 |
| **Training set** | 5,600 images (1,400 per class) |
| **Testing set** | 1,600 images (400 per class) |
| **Validation set** | Created from Training (15% split → ~840 images) |
| **Format** | JPG, grayscale MRI scans |

The dataset comes pre-split into `Training/` and `Testing/` folders, each with a subfolder for each class. A validation split is generated programmatically from the Training folder (85% training and 15% validation). The Testing folder is kept aside and used only for the final evaluation.

**Label distribution:** The dataset is balanced across all four classes, with 1,400 images each in Training and 400 each in Testing. This is illustrated in the bar chart created in the notebook.

**Sample images:** The notebook's EDA section shows one representative image per class.

---

## 2. Model Architecture

- **Base model:** EfficientNetB0, pretrained on ImageNet (include_top=False)
- **Custom head:** GlobalAveragePooling2D → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.2) → Dense(4, Softmax)
- **Training strategy:**
  1. Phase 1: Freeze the base model and only train the custom head (Adam, lr=1e-4)
  2. Phase 2: Unfreeze the last 20 layers of EfficientNetB0 and train at a lower learning rate (Adam, lr=1e-5)
- **Regularization:** Dropout layers, EarlyStopping, and ReduceLROnPlateau
- **Input size:** 224 × 224 × 3
- **Loss function:** Categorical Crossentropy
- **Metrics tracked:** Accuracy, AUC

---

## 3. Evaluation

The trained model is evaluated on the held-out **Testing set** (1,600 images that the model has not seen during training) using:

- **Accuracy, Precision, Recall, F1-Score** via `classification_report` (overall and per class)
- **Confusion Matrix** to visualize per-class prediction errors
- **ROC Curve & AUC** calculated one-vs-rest per class (multi-class setting)
- **Validation AUC** tracked live during training as a Keras metric

Training and validation **Accuracy** and **Loss** curves are plotted across all epochs to show learning progress and check for overfitting.

### Final Test Set Results (1,600 held-out images)

| Metric | Value |
|---|---|
| **Overall Test Accuracy** | 81.94% |
| **Macro Avg Precision** | 82.32% |
| **Macro Avg Recall** | 81.94% |
| **Macro Avg F1-Score** | 81.13% |
| **Macro Avg AUC** | 0.9625 |

**Per-class breakdown:**

| Class | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|
| Glioma | 87.80% | 64.75% | 74.54% | 0.92 |
| Meningioma | 79.34% | 66.25% | 72.22% | 0.94 |
| No Tumor | 82.50% | 99.00% | 90.00% | 1.00 |
| Pituitary | 79.63% | 97.75% | 87.76% | 0.99 |

**Observations:** No Tumor and Pituitary classes are identified with very high accuracy (AUC around 0.99 to 1.00). Glioma and Meningioma show more overlap with each other (indicated by higher off-diagonal counts in the confusion matrix). This is a known challenge in brain MRI classification since these two tumor types can look similar in certain scan planes. This is a good area for future improvement, such as further fine-tuning, more training epochs, or additional data augmentation focused on these two classes.

---

## 4. AI-Assisted Medical Report (LLM Integration)

After each prediction, the app sends the predicted class and confidence score to **Google's Gemini API** (gemini-1.5-flash). This generates a short, plain-language explanatory report that includes:

1. What the predicted class means
2. Commonly linked symptoms (general educational information)
3. A recommendation to consult a qualified radiologist or neurologist

This report is shown in the app and stored in the database alongside the prediction.

**Important:** This is clearly labeled as educational and informational, not as a diagnostic tool — refer to the Disclaimer section below.

---

## 5. Application (Streamlit + SQLite)

The Streamlit app (app.py) offers:

- MRI image upload and preview
- One-click prediction with confidence score and full probability breakdown
- AI-generated medical report for each prediction
- A prediction history table (SQLite-backed), which includes:
  - Total predictions and the last prediction shown in the sidebar
  - Downloadable CSV export
  - Clear-history option
- A disclaimer footer

**Modular file structure:**

```
Brain_MRI_Project/
│
├── app.py                     # Streamlit UI (entry point)
├── predict.py                 # Model loading and image prediction
├── database.py                # SQLite storage for prediction history
├── medical_report.py          # Gemini LLM report generation
│
├── brain_tumor_model.keras    # Trained model (from training notebook)
├── class_names.json           # Class label order (from training notebook)
├── predictions.db             # SQLite DB (created automatically at runtime)
│
├── requirements.txt
├── README.md
│
├── Training/                  # Dataset (Training split)
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
├── Testing/                   # Dataset (Testing split)
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Brain_MRI_Training.ipynb   # Training notebook (data to trained model)
```

---

## 6. Setup & Running Instructions

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Set your Gemini API key
```bash
# macOS / Linux
export GEMINI_API_KEY="your_key_here"

# Windows (PowerShell)
$env:GEMINI_API_KEY="your_key_here"
```
(See the API key setup guide for instructions on obtaining a key from Google AI Studio.)

### Step 3 — Train the model
Run all cells in `Brain_MRI_Training.ipynb`. This will produce:
- `brain_tumor_model.keras`
- `class_names.json`

Place both files in the project root, next to `app.py`.

### Step 4 — Run the Streamlit app
```bash
streamlit run app.py
```
Visit the local URL shown in the terminal (usually `http://localhost:8501`).

---

## 7. Application Flow

```
Upload MRI Image
      ↓
   app.py (UI)
      ↓
  predict.py → brain_tumor_model.keras
      ↓
  Prediction and Confidence
      ↓
medical_report.py → Gemini AI
      ↓
  database.py → SQLite (predictions.db)
      ↓
Displayed in History Table and Downloadable CSV
```

---

## 8. Deliverables Checklist

| Requirement | Status | Location |
|---|---|---|
| Dataset selection and source | ✅ | Section 1 above |
| Dataset stats, train/val/test split | ✅ | Section 1 / Notebook Step 2 |
| Label distribution | ✅ | Notebook Step 3 |
| 5+ sample images | ✅ | Notebook Step 4 |
| EfficientNetB0 transfer learning model | ✅ | Notebook Step 6 |
| Accuracy/Loss training curves | ✅ | Notebook Step 10 |
| Accuracy, Precision, Recall, F1 | ✅ | Notebook Step 12 |
| ROC Curve, AUC, Validation AUC | ✅ | Notebook Step 14 |
| Confusion Matrix | ✅ | Notebook Step 13 |
| Gemini AI-assisted medical report | ✅ | `medical_report.py` |
| Streamlit image upload and prediction | ✅ | `app.py`, `predict.py` |
| SQLite prediction history | ✅ | `database.py` |
| Complete source code | ✅ | All `.py` files |
| Trained model | ✅ | `brain_tumor_model.keras` |
| README/documentation | ✅ | This file |
| `requirements.txt` | ✅ | Included |

---

## 9. Disclaimer

This project is for **educational and research purposes only**. Predictions and AI-generated reports are created by a machine learning model and an LLM. They must **not** be considered a medical diagnosis. Always consult a qualified radiologist or neurologist for professional evaluation.