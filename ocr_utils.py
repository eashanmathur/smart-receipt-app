import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import re

# ===============================
#  STEP 1 — Read Image & Extract Text (Improved OCR)
# ===============================
def read_image_text(image_path):
    """
    Improved OCR with preprocessing — enhances accuracy across receipts.
    Works better for Zomato, Swiggy, Flipkart, Ola, etc.
    """
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise and smooth
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Threshold for better contrast
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup (remove small specks)
    kernel = np.ones((1, 1), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # OCR with tuned config
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_blacklist=@#$%^*_~'
    text = pytesseract.image_to_string(clean, config=custom_config)

    return text


# ===============================
#  STEP 2 — Normalize OCR Text
# ===============================
def normalize_ocr_text(text):
    """
    Fixes common OCR spelling and spacing mistakes.
    """
    replacements = {
        "ree Hundred": "Three Hundred",
        "Rupees And Fifty": "Rupees and Fifty",
        "Buryer": "Burger",
        "Restaurart": "Restaurant",
        "INR ": "₹",
        "Rupees Only": "₹",
        "  ": " "
    }

    for wrong, right in replacements.items():
        text = text.replace(wrong, right)

    # Remove weird non-ASCII characters
    text = ''.join(ch for ch in text if ord(ch) < 128)
    return text


# ===============================
# STEP 3 — Extract Total Amount
# ===============================
import re

def extract_amount(text):
    """
    💰 Final OCR Amount Extractor
    ✅ Fixes 71869 → 1869 error
    ✅ Works for Zomato, Swiggy, Amazon, CRED, Dineout, etc.
    ✅ Keeps context-based keyword matching priority
    """

    # Clean text
    text = text.replace(",", "").replace("₹", " ").replace("?", "").replace("=", "")
    text = re.sub(r"\s+", " ", text)
    text = text.lower()

    # Fix OCR errors like '71869' → '1869'
    text = re.sub(r"\b[56]1(\d{3,5}\.\d{1,2})", r"\1", text)  # removes weird leading '51', '61', '71'
    text = re.sub(r"\b7(\d{3,4}\.\d{1,2})", r"\1", text)      # removes single stray '7' prefix before thousands

    # Fix decimals like "1533 30" → "1533.30"
    text = re.sub(r"\b(\d{2,6})\s(\d{1,2})\b", r"\1.\2", text)

    # 1️⃣ Handle "You Paid" / "Total Paid"
    match = re.search(r"(you\s+paid|total\s+paid)[^0-9]{0,10}(\d{2,6}\.\d{1,3})", text)
    if match:
        return round(float(match.group(2)), 2), "You Paid / Total Paid"

    # 2️⃣ Handle "Bill Amount" / "Total Amount"
    match = re.search(r"(bill\s+amount|total\s+amount|grand\s+total|net\s+amount)[^0-9]{0,10}(\d{2,6}\.\d{1,3})", text)
    if match:
        return round(float(match.group(2)), 2), "Bill / Total Amount"

    # 3️⃣ Handle "Amount of" (Zomato-style)
    match = re.search(r"amount\s+of\s+(\d{2,6}\.\d{1,3})", text)
    if match:
        return round(float(match.group(1)), 2), "Zomato-style"

    # 4️⃣ Amazon/Flipkart — Before "Amount in Words"
    if "amount in words" in text:
        context = text[: text.find("amount in words")]
        nums = re.findall(r"(\d{3,6}\.\d{1,3})", context[-200:])
        nums = [float(n) for n in nums if not re.search(r"(igst|sgst|cgst|tax|discount)", context)]
        if nums:
            return round(max(nums), 2), "Amazon-style (near Amount in Words)"

    # 5️⃣ Final fallback (choose most realistic number)
    all_nums = [float(n) for n in re.findall(r"\d{2,6}\.\d{1,3}", text)]
    valid = [v for v in all_nums if 50 <= v <= 100000]

    # Prefer mid-range values (ignore too large like 67727)
    filtered = [v for v in valid if v <= 10000]
    if filtered:
        return round(max(filtered), 2), "Fallback-filtered"
    elif valid:
        return round(min(valid), 2), "Fallback-min-valid"

    return None, "Not detected"
