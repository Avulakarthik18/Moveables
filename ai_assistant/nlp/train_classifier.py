"""
══════════════════════════════════════════════════════════════════════════
  ROADMIND AI — ML CLASSIFIER TRAINING SCRIPT
  File: ai_assistant/nlp/train_classifier.py

  RUN ONCE to train and save the Random Forest models:
      python ai_assistant/nlp/train_classifier.py

  Re-run if you change category definitions or want to retrain.
  REQUIRES: pip install scikit-learn joblib numpy
══════════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

NLP_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
#  DATASET 1: CAR BOOKING
#  Features: [duration_days, car_type, with_driver, outstation, budget_range]
#  Labels:   0=Economy, 1=Standard, 2=Premium, 3=WithDriverStandard, 4=Luxury
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  TRAINING CAR BOOKING CLASSIFIER")
print("=" * 60)

# fmt: off
car_data = [
    # [duration_days, car_type, with_driver, outstation, budget_range]  → label
    # Economy (0)
    [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0],
    [1, 0, 0, 1, 0], [3, 0, 0, 0, 0], [2, 0, 0, 0, 0],
    [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0],
    [1, 0, 0, 1, 0], [3, 0, 0, 0, 0], [1, 0, 0, 0, 0],
    # Standard (1)
    [2, 1, 0, 0, 1], [3, 1, 0, 0, 1], [1, 1, 0, 0, 1],
    [5, 1, 0, 0, 0], [3, 0, 0, 1, 1], [4, 1, 0, 0, 1],
    [2, 1, 0, 0, 1], [3, 1, 0, 0, 1], [1, 1, 0, 0, 1],
    [5, 1, 0, 0, 0], [3, 0, 0, 1, 1], [4, 1, 0, 0, 1],
    # Premium (2)
    [3, 2, 0, 1, 1], [5, 2, 0, 0, 1], [7, 1, 0, 1, 1],
    [4, 2, 0, 0, 2], [3, 2, 0, 1, 1], [6, 2, 0, 0, 1],
    [3, 2, 0, 1, 1], [5, 2, 0, 0, 1], [7, 1, 0, 1, 1],
    [4, 2, 0, 0, 2], [3, 2, 0, 1, 1], [6, 2, 0, 0, 1],
    # With Driver Standard (3)
    [1, 1, 1, 0, 1], [3, 1, 1, 1, 1], [2, 0, 1, 0, 0],
    [1, 1, 1, 0, 0], [4, 1, 1, 1, 1], [2, 1, 1, 0, 1],
    [1, 1, 1, 0, 1], [3, 1, 1, 1, 1], [2, 0, 1, 0, 0],
    [1, 1, 1, 0, 0], [4, 1, 1, 1, 1], [2, 1, 1, 0, 1],
    # Luxury (4)
    [1, 3, 1, 0, 2], [3, 3, 1, 1, 2], [2, 3, 0, 0, 2],
    [1, 2, 1, 0, 2], [2, 3, 1, 1, 2], [1, 3, 1, 0, 2],
    [1, 3, 1, 0, 2], [3, 3, 1, 1, 2], [2, 3, 0, 0, 2],
    [1, 2, 1, 0, 2], [2, 3, 1, 1, 2], [1, 3, 1, 0, 2],
]
car_labels = (
    [0]*12 + [1]*12 + [2]*12 + [3]*12 + [4]*12
)
# fmt: on

X_car = np.array(car_data)
y_car = np.array(car_labels)

X_train, X_test, y_train, y_test = train_test_split(X_car, y_car, test_size=0.2, random_state=42)

car_clf = RandomForestClassifier(n_estimators=100, random_state=42)
car_clf.fit(X_train, y_train)

print(f"  Training samples: {len(X_train)}  |  Test samples: {len(X_test)}")
print(f"  Accuracy: {car_clf.score(X_test, y_test) * 100:.1f}%")
print()
print(classification_report(
    y_test, car_clf.predict(X_test),
    target_names=["Economy", "Standard", "Premium", "WithDriver", "Luxury"]
))

car_path = os.path.join(NLP_DIR, "car_booking_model.pkl")
joblib.dump(car_clf, car_path)
print(f"✅ Car booking model saved → {car_path}")
print()


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET 2: PARCEL RISK CLASSIFIER
#  Features: [item_type, weight_kg_bucket, distance, urgency]
#  Labels:   0=Standard, 1=ExpressInsurance, 2=SpecialHandling, 3=HighRisk
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  TRAINING PARCEL RISK CLASSIFIER")
print("=" * 60)

# fmt: off
parcel_data = [
    # [item_type, weight, distance, urgency] → label
    # Standard (0) — documents, light, local, not urgent
    [0, 0, 0, 0], [0, 0, 1, 0], [3, 0, 0, 0],
    [3, 1, 0, 0], [5, 0, 0, 0], [0, 1, 0, 0],
    [0, 0, 0, 0], [0, 0, 1, 0], [3, 0, 0, 0],
    [3, 1, 0, 0], [5, 0, 0, 0], [0, 1, 0, 0],
    # Express with Insurance (1) — electronics, medium, or urgent
    [1, 1, 1, 1], [1, 0, 2, 1], [3, 1, 1, 1],
    [0, 1, 2, 1], [1, 1, 0, 1], [5, 1, 1, 1],
    [1, 1, 1, 1], [1, 0, 2, 1], [3, 1, 1, 1],
    [0, 1, 2, 1], [1, 1, 0, 1], [5, 1, 1, 1],
    # Special Handling (2) — fragile, food/perishable
    [2, 0, 0, 0], [2, 1, 1, 0], [4, 0, 0, 0],
    [2, 0, 2, 0], [4, 1, 0, 0], [2, 1, 0, 1],
    [2, 0, 0, 0], [2, 1, 1, 0], [4, 0, 0, 0],
    [2, 0, 2, 0], [4, 1, 0, 0], [2, 1, 0, 1],
    # High Risk (3) — very heavy, or extremely large
    [5, 3, 2, 0], [0, 3, 1, 0], [1, 3, 2, 1],
    [3, 3, 0, 0], [2, 3, 2, 1], [5, 3, 1, 0],
    [5, 3, 2, 0], [0, 3, 1, 0], [1, 3, 2, 1],
    [3, 3, 0, 0], [2, 3, 2, 1], [5, 3, 1, 0],
]
parcel_labels = (
    [0]*12 + [1]*12 + [2]*12 + [3]*12
)
# fmt: on

X_parcel = np.array(parcel_data)
y_parcel = np.array(parcel_labels)

X_train, X_test, y_train, y_test = train_test_split(X_parcel, y_parcel, test_size=0.2, random_state=42)

parcel_clf = RandomForestClassifier(n_estimators=100, random_state=42)
parcel_clf.fit(X_train, y_train)

print(f"  Training samples: {len(X_train)}  |  Test samples: {len(X_test)}")
print(f"  Accuracy: {parcel_clf.score(X_test, y_test) * 100:.1f}%")
print()
print(classification_report(
    y_test, parcel_clf.predict(X_test),
    target_names=["Standard", "Express+Insurance", "SpecialHandling", "HighRisk"]
))

parcel_path = os.path.join(NLP_DIR, "parcel_model.pkl")
joblib.dump(parcel_clf, parcel_path)
print(f"✅ Parcel model saved → {parcel_path}")
print()
print("🎉 Both models trained and saved. The ML flow is now live!")
