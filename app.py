import streamlit as st
import os
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from ocr_utils import read_image_text, extract_amount, normalize_ocr_text
import matplotlib.pyplot as plt

# ===============================
#  Load model and label encoder
# ===============================
model = load_model("model/expense_model.h5")
with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ===============================
#  Page Setup
# ===============================
st.set_page_config(page_title="Smart Receipt App", page_icon="🧾", layout="wide")
st.markdown("<h1 style='color:#4CAF50;text-align:center;'>🧾 Smart Receipt Scanner & Expense Dashboard</h1>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📤 Upload Receipt", "📊 Expense Dashboard"])

# ===============================
# TAB 1 — Upload and Categorize
# ===============================
with tab1:
    uploaded_file = st.file_uploader("Upload your receipt image", type=["jpg","jpeg","png"])
    
    if uploaded_file:
        os.makedirs("data/saved_receipts", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join("data/saved_receipts", f"{timestamp}_{uploaded_file.name}")
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(save_path, caption="🧾 Uploaded Receipt", use_column_width=True)
        st.info("🔍 Reading text...")

        # Step 1 — OCR
        text = read_image_text(save_path)
        # Step 2 — Clean OCR
        text = normalize_ocr_text(text)

        st.subheader("📜 Extracted Text")
        st.text_area("Extracted OCR Text", text, height=180)

        # Step 3 — Extract Amount
        detected_amount, match_source = extract_amount(text)
        if detected_amount:
            st.success(f"💰 Detected Total Amount: ₹{detected_amount}")
            st.caption(f"🔍 Matched via: {match_source}")
            total_amount = st.number_input("Total Bill Amount (₹)", value=float(detected_amount), step=0.01)
        else:
            st.warning("⚠️ Could not detect total amount automatically.")
            total_amount = st.number_input("Total Bill Amount (₹)", min_value=0.0, step=0.01)

        num_people = st.number_input("People Paid For", min_value=1, max_value=10, value=1)

        if st.button("🧠 Categorize & Save"):
            # --- Existing ML model classification ---
            tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
            tokenizer.fit_on_texts([text])
            seq = tokenizer.texts_to_sequences([text])
            padded = pad_sequences(seq, maxlen=20, padding='post', truncating='post')

            prediction = model.predict(padded)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)]) 
            category = predicted_label[0]
            # 🧠 Rule-based Smart Category Correction
            text_lower = text.lower()
            food_keywords = [
                "restaurant", "zomato", "swiggy", "dineout", "barbeque", "romeo lane",
                "cafe", "bar", "pub", "grill", "kitchen", "food", "meals"
                ]
            shopping_keywords = [
                "amazon", "flipkart", "myntra", "ajio", "meesho", "snapdeal",
                "shopping", "order number", "order id", "payment information", "total paid"
                ]
            travel_keywords = [
                "flight", "hotel", "ola", "uber", "trip", "train", "booking", "airbnb"
                ]
            utilities_keywords = [
                "electricity", "water bill", "recharge", "postpaid", "broadband", "gas"
                ]
            # 🔹 Apply in order of confidence
            if any(word in text_lower for word in food_keywords):
                category = "Food & Dining"
            elif any(word in text_lower for word in shopping_keywords):
                category = "Online Shopping"
            elif any(word in text_lower for word in travel_keywords):
                category = "Travel"
            elif any(word in text_lower for word in utilities_keywords):
                category = "Utilities"


            your_share = round(total_amount / num_people, 2) if total_amount > 0 else 0

            st.success(f"💡 Category: **{category}**")
            st.info(f"💸 Your Share: ₹{your_share}")

            with open("data/expenses_log.txt", "a") as log:
                log.write(f"{timestamp},{category},{total_amount},{your_share},{text[:40]}...\n")

            st.success("✅ Expense saved successfully!")

# ===============================
# TAB 2 — Dashboard
# ===============================
with tab2:
    log_path = "data/expenses_log.txt"

    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        df = pd.read_csv(log_path, names=["Timestamp","Category","Total_Amount","Your_Share","Text"])
        
        if not df.empty and "Total_Amount" in df.columns:
            total_spent = df["Total_Amount"].sum()
            st.subheader(f"💰 Total Spent: ₹{round(total_spent, 2)}")

            cat_summary = df.groupby("Category")["Total_Amount"].sum()
            if not cat_summary.empty:
                fig, ax = plt.subplots()
                ax.pie(cat_summary, labels=cat_summary.index, autopct="%1.1f%%", startangle=90)
                st.pyplot(fig)

            st.dataframe(df.tail(10))

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Expense Log", csv, "expenses.csv", "text/csv")
        else:
            st.info("No expenses recorded yet.")
    else:
        st.info("No expense records yet! Upload your first receipt in the 'Upload Receipt' tab.")
