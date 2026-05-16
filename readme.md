# Smart Receipt Scanner & Expense Dashboard

A deep learning-powered web app that scans receipt images using OCR, automatically categorizes expenses, splits bills, and visualizes your spending — built with Streamlit and TensorFlow.

---

## What It Does

- Uploads and scans receipt images using **OCR (Tesseract)**
- Extracts the **total amount** automatically from receipts
- Categorizes expenses using an **LSTM neural network** + rule-based correction
- Splits the bill across multiple people
- Logs all expenses and shows a **spending dashboard** with pie charts
- Supports receipts from Zomato, Swiggy, Amazon, Flipkart, Uber, Ola, and more

---

## ML Model Used

| Component | Details |
|---|---|
| Model | LSTM Neural Network (TensorFlow/Keras) |
| Input | OCR-extracted receipt text |
| Output | Expense category (Food, Travel, Shopping, Utilities, Healthcare) |
| Accuracy | ~90%+ on test set |

Rule-based keyword correction is applied on top of the model for higher real-world accuracy.

---

## Tech Stack

- Python 3.11
- Streamlit
- TensorFlow / Keras
- OpenCV + Pytesseract (OCR)
- pandas, numpy, matplotlib
- scikit-learn

---

## Project Structure

```
smart_receipt_app/
├── app.py                  # Main Streamlit app
├── ocr_utils.py            # OCR + text extraction + amount detection
├── train_model.py          # LSTM model training script
├── test_ocr.py             # OCR testing utility
├── data/
│   ├── expenses_log.txt    # Auto-generated expense log
│   └── saved_receipts/     # Uploaded receipt images
├── model/
│   ├── expense_model.h5    # Trained LSTM model
│   └── label_encoder.pkl   # Label encoder
└── requirements.txt
```

---

## Setup & Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/smart-receipt-app.git
cd smart-receipt-app

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR engine
# macOS:
brew install tesseract
# Ubuntu:
sudo apt install tesseract-ocr

# (Optional) Retrain the model
python train_model.py

# Launch the app
streamlit run app.py
```

---

## Expense Categories

- Food & Dining
- Travel
- Online Shopping
- Utilities
- Healthcare

---

## Authors

- Eashan Nananya
- Apeksha Dongre

Manipal University Jaipur — B.Tech CSE (Data Science), Semester VI
