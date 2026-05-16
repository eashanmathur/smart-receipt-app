import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle
import os

# --- Ensure model directory exists ---
os.makedirs("model", exist_ok=True)

# --- Expanded and more realistic dataset ---
texts = [
    # FOOD
    "Zomato food order 250 rupees",
    "Swiggy lunch bill payment",
    "Burger King restaurant receipt",
    "Pizza Hut dinner 499 bill",
    "KFC takeaway order",
    "Cafe Coffee Day coffee 180",
    "Restaurant dinner payment",
    "Dominos pizza delivery receipt",
    "Barbeque Nation buffet dinner",
    "Subway sandwich lunch receipt",
    "Order from Zomato food delivery",
    "Swiggy breakfast order",
    "Zomato payment receipt for food",
    "McDonalds burger meal",
    "Zomato invoice total 350",

    # TRAVEL
    "Uber cab ride 350",
    "Ola trip to airport",
    "Train ticket booking IRCTC",
    "Flight Indigo ticket 5000",
    "Bus ticket RedBus 250",
    "Cab fare 400",
    "Petrol pump payment 2000",
    "Toll tax highway payment",
    "Ola outstation bill",
    "Flight cancellation fee",
    "Cab fare Uber payment",
    "IRCTC e-ticket confirmed",
    "Flight ticket AirIndia booking",

    # SHOPPING
    "Amazon order for headphones",
    "Flipkart shopping bill 999",
    "Myntra clothes purchase",
    "Big Bazaar grocery shopping 1200",
    "DMart purchase 850",
    "Reliance Trends fashion store",
    "Zudio clothes payment",
    "Croma electronics bill",
    "Decathlon sports shoes purchase",
    "Lifestyle store shopping",
    "Tata Cliq order payment",
    "Nykaa beauty product order",

    # UTILITIES
    "Electricity bill 1450",
    "Water bill payment",
    "Gas cylinder booking receipt",
    "Mobile recharge Jio 399",
    "Airtel broadband bill",
    "Netflix subscription 499",
    "Disney Hotstar renewal charge",
    "Spotify premium 129 payment",
    "DTH TataSky recharge bill",
    "Postpaid Vodafone mobile bill",
    "PhonePe electricity payment",
    "Amazon Prime yearly subscription",

    # HEALTHCARE
    "Pharmacy bill 250",
    "Apollo Hospital consultation",
    "Medical store receipt",
    "Lab test report payment",
    "Doctor appointment 600",
    "Blood test invoice",
    "Eye checkup clinic bill",
    "Dental cleaning payment",
    "Chemist medicine bill",
    "Vaccination charge",
    "MedPlus pharmacy order",
    "Apollo diagnostics lab test"
]

labels = (
    ["Food"] * 15 +
    ["Travel"] * 13 +
    ["Shopping"] * 12 +
    ["Utilities"] * 12 +
    ["Healthcare"] * 12
)

# --- Add extra food-related examples ---
texts += [
    "Zomato food delivery",
    "Zomato receipt",
    "Order from Zomato",
    "Swiggy invoice",
    "Burger King restaurant bill",
    "Restaurant food payment",
    "Cafe bill",
    "Dining restaurant bill",
    "Lunch from Dominos",
    "Dinner from Pizza Hut"
]
labels += ["Food"] * 10

# --- Tokenize text ---
tokenizer = Tokenizer(num_words=3000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=25, padding='post', truncating='post')

# --- Encode labels ---
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

# --- Split into train/test ---
split = int(0.8 * len(padded))
X_train, X_test = padded[:split], padded[split:]
y_train, y_test = encoded_labels[:split], encoded_labels[split:]

# --- Model architecture ---
model = Sequential([
    Embedding(input_dim=3000, output_dim=64, input_length=25),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(set(labels)), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- Train model ---
print("\n🧠 Training expense categorization model...\n")
model.fit(X_train, np.array(y_train), epochs=70, batch_size=4, verbose=1)

# --- Evaluate model ---
print("\n🔍 Evaluating model performance...\n")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='weighted')
recall = recall_score(y_test, y_pred_classes, average='weighted')
f1 = f1_score(y_test, y_pred_classes, average='weighted')

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nDetailed Classification Report:\n")
print(classification_report(y_test, y_pred_classes))

# --- Save model and label encoder ---
model.save("model/expense_model.h5")
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\n✅ Model retrained successfully with expanded dataset!")
