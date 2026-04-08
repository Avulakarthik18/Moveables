"""
══════════════════════════════════════════════════════════════════════════
  ROADMIND AI — NLP INTENT ROUTER (Layer 0)
  File: ai_assistant/nlp/nlp_router.py

  PURPOSE:
    Classifies user message into a structured intent.
    If intent is structured (e.g. "book a car", "send parcel"):
      → routes to ML classifier flow for follow-up questions
    Otherwise:
      → falls through to existing Layer 1-2-3 (DB + RAG + Gemini)

  IMPORTED BY: backend/app.py in the /ai-chat route
══════════════════════════════════════════════════════════════════════════
"""

import re

# ── Intent keyword map ────────────────────────────────────────────────────────
# Each intent maps to a list of trigger phrases (lowercase, partial match OK)
INTENT_MAP = {
    "book_car": [
        "book a car", "rent a car", "hire a car", "need a car",
        "want to rent", "car for rent", "want a car", "book car",
        "i want to book", "can i book", "rental for", "rent for"
    ],
    "send_parcel": [
        "send parcel", "send a parcel", "deliver parcel", "send package",
        "parcel delivery", "courier", "ship a package", "send something",
        "i want to send", "need to deliver", "delivery service"
    ],
    "sell_car": [
        "sell my car", "list my car", "want to sell", "selling my car",
        "i want to sell", "put my car for sale", "list for sale",
        "how to sell", "car for sale"
    ],
    "parcel_classifier": [
        "what type of parcel", "classify my parcel", "is my parcel safe",
        "fragile item", "send fragile", "send electronics",
        "parcel risk", "how to send safely"
    ],
    "price_estimate": [
        "how much does it cost", "what is the price", "price estimate",
        "rental cost", "fare estimate", "how much to rent",
        "what is the fare", "cost of renting", "per day rate"
    ],
}

# ── Intents that should trigger ML classifier conversation flow ───────────────
ML_CLASSIFIER_INTENTS = {"book_car", "send_parcel", "parcel_classifier"}


def detect_intent(message: str) -> str:
    """
    Detects the primary intent of the user message.
    Returns an intent name string, or 'general' if no structured intent found.
    """
    msg_lower = message.lower().strip()

    for intent, triggers in INTENT_MAP.items():
        for trigger in triggers:
            if trigger in msg_lower:
                return intent

    return "general"


def is_ml_intent(intent: str) -> bool:
    """Returns True if the intent should trigger the ML classifier flow."""
    return intent in ML_CLASSIFIER_INTENTS


def get_intent_label(intent: str) -> str:
    """Human-readable label for logging/debugging."""
    labels = {
        "book_car": "Car Booking Request",
        "send_parcel": "Parcel Delivery Request",
        "sell_car": "Car Sell Request",
        "parcel_classifier": "Parcel Risk Classification",
        "price_estimate": "Price Estimate Query",
        "general": "General / Conversational",
    }
    return labels.get(intent, "Unknown")
