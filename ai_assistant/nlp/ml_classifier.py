"""
══════════════════════════════════════════════════════════════════════════
  ROADMIND AI — ML CLASSIFIER (Layer 0b)
  File: ai_assistant/nlp/ml_classifier.py

  PURPOSE:
    - Manages multi-turn follow-up question flows for structured intents
    - Collects feature values through chat conversation
    - Uses a trained Random Forest model to predict the outcome category
    - Returns a structured result that Gemini can use to give a final answer

  USED BY: backend/app.py in the /ai-chat route
  MODELS:  ai_assistant/nlp/car_booking_model.pkl  (trained by train_classifier.py)
           ai_assistant/nlp/parcel_model.pkl        (trained by train_classifier.py)
══════════════════════════════════════════════════════════════════════════
"""

import os
import json
import joblib
import numpy as np
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
NLP_DIR       = os.path.dirname(os.path.abspath(__file__))
CATEGORIES    = json.load(open(os.path.join(NLP_DIR, "categories.json"), encoding="utf-8"))
CAR_MODEL_PATH    = os.path.join(NLP_DIR, "car_booking_model.pkl")
PARCEL_MODEL_PATH = os.path.join(NLP_DIR, "parcel_model.pkl")

_car_model    = None
_parcel_model = None


def _load_car_model():
    global _car_model
    if _car_model is None and os.path.exists(CAR_MODEL_PATH):
        _car_model = joblib.load(CAR_MODEL_PATH)
    return _car_model


def _load_parcel_model():
    global _parcel_model
    if _parcel_model is None and os.path.exists(PARCEL_MODEL_PATH):
        _parcel_model = joblib.load(PARCEL_MODEL_PATH)
    return _parcel_model


# ─────────────────────────────────────────────────────────────────────────────
#  CAR BOOKING FLOW
# ─────────────────────────────────────────────────────────────────────────────

CAR_QUESTIONS = [
    {
        "key":      "duration_days",
        "question": "How many days do you need the car? 📅\n(e.g., 1 day, 3 days, 1 week)",
        "parse":    lambda ans: _parse_duration(ans),
    },
    {
        "key":      "car_type",
        "question": "What type of car are you looking for? 🚗\n1️⃣ Hatchback (small, budget)\n2️⃣ Sedan (comfortable)\n3️⃣ SUV (spacious)\n4️⃣ Luxury (premium)\nType 1, 2, 3, or 4 — or just describe it!",
        "parse":    lambda ans: _parse_car_type(ans),
    },
    {
        "key":      "with_driver",
        "question": "Do you need a driver, or will you drive yourself? 🧑‍✈️\n1️⃣ I'll drive myself (Rental Only)\n2️⃣ I need a driver (With Driver)",
        "parse":    lambda ans: 1 if any(w in ans.lower() for w in ["driver", "2", "need driver", "with driver"]) else 0,
    },
    {
        "key":      "outstation",
        "question": "Is this for a city trip or outstation travel? 🗺️\n1️⃣ City / local\n2️⃣ Outstation / long distance",
        "parse":    lambda ans: 1 if any(w in ans.lower() for w in ["outstation", "long", "2", "outside city", "station", "highway"]) else 0,
    },
    {
        "key":      "budget_range",
        "question": "What's your rough budget per day? 💰\n1️⃣ Budget (under ₹1,500/day)\n2️⃣ Mid-range (₹1,500–₹3,000/day)\n3️⃣ Premium (above ₹3,000/day)",
        "parse":    lambda ans: _parse_budget(ans),
    },
]

PARCEL_QUESTIONS = [
    {
        "key":      "item_type",
        "question": "What are you sending? 📦\n1️⃣ Documents / files\n2️⃣ Electronics (phone, laptop, etc.)\n3️⃣ Fragile items (glassware, crockery)\n4️⃣ Clothing / personal items\n5️⃣ Food / perishables\n6️⃣ Other",
        "parse":    lambda ans: _parse_item_type(ans),
    },
    {
        "key":      "weight_kg",
        "question": "Approximate weight of the parcel? ⚖️\n1️⃣ Light (under 2 kg)\n2️⃣ Medium (2–5 kg)\n3️⃣ Heavy (5–15 kg)\n4️⃣ Very heavy (over 15 kg)",
        "parse":    lambda ans: _parse_weight(ans),
    },
    {
        "key":      "distance",
        "question": "What's the delivery distance? 📍\n1️⃣ Local / same city\n2️⃣ Nearby city / outstation\n3️⃣ Interstate / long distance",
        "parse":    lambda ans: _parse_distance(ans),
    },
    {
        "key":      "urgency",
        "question": "How urgent is this delivery? ⏱️\n1️⃣ Standard (flexible timing)\n2️⃣ Express (need it fast!)",
        "parse":    lambda ans: 1 if any(w in ans.lower() for w in ["express", "urgent", "fast", "quick", "asap", "2"]) else 0,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  PARSE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_duration(text):
    text = text.lower()
    import re
    nums = re.findall(r'\d+', text)
    if nums:
        n = int(nums[0])
        if "week" in text: n *= 7
        if "month" in text: n *= 30
        return min(n, 30)
    if "day" in text:   return 1
    if "week" in text:  return 7
    if "month" in text: return 30
    return 1


def _parse_car_type(text):
    text = text.lower()
    if any(w in text for w in ["luxury", "4", "premium", "bmw", "mercedes", "audi"]): return 3
    if any(w in text for w in ["suv", "3", "spacious", "fortuner", "creta", "nexon"]): return 2
    if any(w in text for w in ["sedan", "2", "comfortable", "city", "honda", "verna"]): return 1
    return 0  # hatchback / default


def _parse_budget(text):
    text = text.lower()
    import re
    nums = re.findall(r'\d+', text)
    if nums:
        n = int(nums[0])
        if n < 1500:  return 0
        if n < 3000:  return 1
        return 2
    if any(w in text for w in ["1", "budget", "cheap", "low", "under"]): return 0
    if any(w in text for w in ["2", "mid", "medium", "moderate"]):        return 1
    if any(w in text for w in ["3", "premium", "high", "luxury"]):        return 2
    return 1


def _parse_item_type(text):
    text = text.lower()
    if any(w in text for w in ["document", "file", "paper", "1"]):           return 0
    if any(w in text for w in ["electronic", "phone", "laptop", "2"]):       return 1
    if any(w in text for w in ["fragile", "glass", "crockery", "3"]):        return 2
    if any(w in text for w in ["cloth", "personal", "garment", "4"]):        return 3
    if any(w in text for w in ["food", "perishable", "vegetable", "5"]):     return 4
    return 5  # other


def _parse_weight(text):
    text = text.lower()
    import re
    nums = re.findall(r'\d+', text)
    if nums:
        n = int(nums[0])
        if n < 2:   return 0
        if n < 5:   return 1
        if n < 15:  return 2
        return 3
    if any(w in text for w in ["1", "light", "small"]):  return 0
    if any(w in text for w in ["2", "medium"]):           return 1
    if any(w in text for w in ["3", "heavy"]):            return 2
    if any(w in text for w in ["4", "very heavy"]):       return 3
    return 1


def _parse_distance(text):
    text = text.lower()
    if any(w in text for w in ["1", "local", "city", "same"]): return 0
    if any(w in text for w in ["2", "nearby", "outstation"]):  return 1
    if any(w in text for w in ["3", "interstate", "long"]):    return 2
    return 0


# ─────────────────────────────────────────────────────────────────────────────
#  PREDICTION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

CAR_CATEGORIES   = ["Economy Rental", "Standard Rental", "Premium Rental", "With Driver Standard", "Luxury Package"]
PARCEL_CATEGORIES = ["Standard Delivery", "Express with Insurance", "Special Handling Required", "High Risk — Contact Support"]


def predict_car_category(features: dict) -> str:
    """Given collected features, predict the best car rental category."""
    model = _load_car_model()
    if model:
        X = np.array([[
            features.get("duration_days", 1),
            features.get("car_type", 0),
            features.get("with_driver", 0),
            features.get("outstation", 0),
            features.get("budget_range", 1),
        ]])
        idx = model.predict(X)[0]
        return CAR_CATEGORIES[int(idx)]
    # Rule-based fallback when model not trained yet
    return _rule_based_car(features)


def predict_parcel_category(features: dict) -> str:
    """Given collected features, predict the parcel handling category."""
    model = _load_parcel_model()
    if model:
        X = np.array([[
            features.get("item_type", 5),
            features.get("weight_kg", 1),
            features.get("distance", 0),
            features.get("urgency", 0),
        ]])
        idx = model.predict(X)[0]
        return PARCEL_CATEGORIES[int(idx)]
    return _rule_based_parcel(features)


# ─────────────────────────────────────────────────────────────────────────────
#  RULE-BASED FALLBACKS (no model needed — works immediately)
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_car(f: dict) -> str:
    with_driver  = f.get("with_driver", 0)
    car_type     = f.get("car_type", 0)
    budget       = f.get("budget_range", 1)
    days         = f.get("duration_days", 1)

    if with_driver == 1:
        if car_type == 3 or budget == 2:
            return "Luxury Package"
        return "With Driver Standard"
    if car_type == 3 or budget == 2:
        return "Luxury Package"
    if car_type == 2 or (days >= 3 and budget == 1):
        return "Premium Rental"
    if car_type == 1 or budget == 1:
        return "Standard Rental"
    return "Economy Rental"


def _rule_based_parcel(f: dict) -> str:
    item_type = f.get("item_type", 5)   # 0=doc,1=electronics,2=fragile,3=cloth,4=food,5=other
    weight    = f.get("weight_kg", 1)   # 0=light,1=medium,2=heavy,3=very heavy
    urgency   = f.get("urgency", 0)     # 0=standard, 1=express

    if weight == 3:                             return "High Risk — Contact Support"
    if item_type == 2 or item_type == 4:        return "Special Handling Required"
    if item_type == 1 or urgency == 1:          return "Express with Insurance"
    return "Standard Delivery"


# ─────────────────────────────────────────────────────────────────────────────
#  FLOW STATE MANAGER
# ─────────────────────────────────────────────────────────────────────────────

def get_next_question(flow_type: str, collected: dict) -> Optional[dict]:
    """
    Given the flow type and already-collected features, returns
    the next question dict {'key', 'question'} or None if all answers collected.
    """
    questions = CAR_QUESTIONS if flow_type == "book_car" else PARCEL_QUESTIONS
    for q in questions:
        if q["key"] not in collected:
            return q
    return None  # all questions answered


def parse_answer(flow_type: str, key: str, answer: str):
    """
    Parses a user's answer for the given question key.
    Returns the parsed/encoded value.
    """
    questions = CAR_QUESTIONS if flow_type == "book_car" else PARCEL_QUESTIONS
    for q in questions:
        if q["key"] == key:
            try:
                return q["parse"](answer)
            except:
                return None
    return None


def build_result_message(flow_type: str, category: str) -> str:
    """Builds a friendly result message with the category info and workflow steps."""
    cats = CATEGORIES.get(
        "car_booking" if flow_type == "book_car" else "parcel_delivery",
        {}
    )
    info = cats.get(category, {})

    if flow_type == "book_car":
        return (
            f"🎉 **Awesome! Based on your needs, I recommend the {category} tier!** 🚗\n\n"
            f"**What it is:** {info.get('description', '')}\n"
            f"**Typical cars:** {info.get('examples', '')}\n"
            f"**Price Range:** {info.get('typical_price', '')}\n\n"
            f"### 📋 Your Next Steps to Book:\n"
            f"1️⃣ **Find Your Car:** Go to the Home Page and look for cars in this category.\n"
            f"2️⃣ **Choose Dates:** Select your pickup and drop-off dates.\n"
            f"3️⃣ **Payment:** Pay securely via Razorpay (Security deposit applies for self-drive).\n"
            f"4️⃣ **Pickup:** Meet the owner/driver and show your ID!\n\n"
            f"*(I will pull up some currently available live cars for you below 👇)*"
        )
    else:
        return (
            f"🎉 **Great! Your parcel falls under the '{category}' tier!** 📦\n\n"
            f"**What this means:** {info.get('description', '')}\n"
            f"**How it's handled:** {info.get('handling', '')}\n\n"
            f"### 📋 Your Next Steps to Send:\n"
            f"1️⃣ **Create Request:** Go to *Services → Send Parcel* and fill the form.\n"
            f"2️⃣ **Wait for Driver:** A verified driver will accept your request.\n"
            f"3️⃣ **Handover (QR):** The driver will scan your 12-digit QR code at pickup.\n"
            f"4️⃣ **Delivery (OTP):** The receiver must give the 4-digit OTP to complete delivery.\n\n"
            f"Congratulations on taking the first step! Let me know if you need help tracking it."
        )
