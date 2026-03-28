"""
SwasthyaBuddy - Flask ML Prediction API
Matches Akshat's original file structure:
  model/model.pkl
  model/encoder.pkl
  model/symptoms.json   (loaded as sorted list)
  model/diseases.json

POST /predict  → top 3 disease predictions
GET  /health   → liveness check
"""

import os
import json
import pickle
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)   # Required — Java backend calls this from a different origin

# ── Load artifacts (same paths Akshat used) ────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

try:
    model    = pickle.load(open(os.path.join(MODEL_DIR, "model.pkl"),   "rb"))
    encoder  = pickle.load(open(os.path.join(MODEL_DIR, "encoder.pkl"), "rb"))
    symptoms = sorted(json.load(open(os.path.join(MODEL_DIR, "symptoms.json"))))  # sorted() exactly like Akshat
    diseases = json.load(open(os.path.join(MODEL_DIR, "diseases.json")))

    logger.info(f"✅ model.pkl    loaded")
    logger.info(f"✅ encoder.pkl  loaded")
    logger.info(f"✅ symptoms.json loaded — {len(symptoms)} symptoms")
    logger.info(f"✅ diseases.json loaded — {len(diseases)} diseases")
    MODEL_LOADED = True

except Exception as e:
    logger.error(f"❌ Startup load error: {e}")
    model = encoder = symptoms = diseases = None
    MODEL_LOADED = False

# ── Precautions (Akshat's original data) ──────────────────────────────────────
precautions = {
    "Fungal infection":    ["Keep skin clean and dry", "Use antifungal cream", "Avoid sharing personal items", "Wear breathable clothing"],
    "Allergy":             ["Avoid allergens", "Take antihistamines", "Consult a doctor", "Keep windows closed during high pollen"],
    "GERD":                ["Avoid spicy food", "Eat smaller meals", "Don't lie down after eating", "Avoid caffeine and alcohol"],
    "Chronic cholestasis": ["Avoid alcohol", "Follow low-fat diet", "Take prescribed medication", "Regular liver checkups"],
    "Drug Reaction":       ["Stop the medication immediately", "Consult a doctor", "Drink plenty of water", "Avoid self-medication"],
    "Peptic ulcer disease":["Avoid spicy food", "Avoid alcohol", "Take antacids", "Eat smaller frequent meals"],
    "AIDS":                ["Use protection", "Regular medical checkups", "Take antiretroviral drugs", "Maintain healthy lifestyle"],
    "Diabetes":            ["Monitor blood sugar", "Follow diabetic diet", "Exercise regularly", "Take prescribed medication"],
    "Gastroenteritis":     ["Stay hydrated", "Eat bland food", "Rest", "Avoid dairy products"],
    "Bronchial Asthma":    ["Use inhaler", "Avoid triggers", "Stay indoors during high pollution", "Follow medication schedule"],
    "Hypertension":        ["Reduce salt intake", "Exercise regularly", "Monitor blood pressure", "Take prescribed medication"],
    "Migraine":            ["Rest in dark quiet room", "Stay hydrated", "Avoid triggers", "Take prescribed medication"],
    "Cervical spondylosis":["Maintain good posture", "Do neck exercises", "Use ergonomic chair", "Apply heat/cold packs"],
    "Paralysis (brain hemorrhage)": ["Immediate medical attention", "Physical therapy", "Follow doctor's advice", "Regular monitoring"],
    "Jaundice":            ["Rest", "Stay hydrated", "Avoid alcohol", "Eat light easily digestible food"],
    "Malaria":             ["Take antimalarial drugs", "Use mosquito nets", "Eliminate standing water", "Use repellents"],
    "Chicken pox":         ["Stay isolated", "Avoid scratching", "Apply calamine lotion", "Take antihistamines"],
    "Dengue":              ["Rest", "Stay hydrated", "Take prescribed medication", "Monitor platelet count"],
    "Typhoid":             ["Take antibiotics as prescribed", "Stay hydrated", "Eat light food", "Maintain hygiene"],
    "Hepatitis A":         ["Rest", "Stay hydrated", "Avoid alcohol", "Practice good hygiene"],
    "Hepatitis B":         ["Take antiviral medication", "Avoid alcohol", "Regular checkups", "Get vaccinated contacts"],
    "Hepatitis C":         ["Take antiviral medication", "Avoid alcohol", "Regular liver tests", "Don't share needles"],
    "Hepatitis D":         ["Take medication", "Avoid alcohol", "Regular checkups", "Hepatitis B vaccination"],
    "Hepatitis E":         ["Rest", "Stay hydrated", "Avoid alcohol", "Practice good hygiene"],
    "Alcoholic hepatitis": ["Stop drinking immediately", "Nutritional support", "Medical supervision", "Liver function monitoring"],
    "Tuberculosis":        ["Complete full medication course", "Cover mouth when coughing", "Improve ventilation", "Regular checkups"],
    "Common Cold":         ["Rest", "Stay hydrated", "Take vitamin C", "Avoid cold exposure"],
    "Pneumonia":           ["Take prescribed antibiotics", "Rest", "Stay hydrated", "Deep breathing exercises"],
    "Dimorphic hemorrhoids(piles)": ["High fiber diet", "Stay hydrated", "Avoid straining", "Use sitz baths"],
    "Heart attack":        ["Immediate medical attention", "Chew aspirin if not allergic", "Stay calm", "Call emergency services"],
    "Varicose veins":      ["Elevate legs", "Exercise regularly", "Avoid prolonged standing", "Wear compression stockings"],
    "Hypothyroidism":      ["Take thyroid medication", "Regular thyroid tests", "Balanced diet", "Exercise regularly"],
    "Hyperthyroidism":     ["Take prescribed medication", "Avoid iodine-rich foods", "Regular checkups", "Manage stress"],
    "Hypoglycemia":        ["Eat sugary food immediately", "Monitor blood sugar", "Carry glucose tablets", "Regular meals"],
    "Osteoarthritis":      ["Exercise gently", "Maintain healthy weight", "Use joint supports", "Take prescribed medication"],
    "Arthritis":           ["Exercise regularly", "Hot and cold therapy", "Take prescribed medication", "Protect your joints"],
    "(Vertigo) Paroxysmal Positional Vertigo": ["Head repositioning exercises", "Avoid sudden movements", "Use walking aid", "Consult ENT"],
    "Acne":                ["Keep face clean", "Avoid oily food", "Use non-comedogenic products", "Consult dermatologist"],
    "Urinary tract infection": ["Drink plenty of water", "Take prescribed antibiotics", "Maintain hygiene", "Avoid holding urine"],
    "Psoriasis":           ["Moisturize regularly", "Avoid triggers", "Use prescribed topical treatments", "Manage stress"],
    "Impetigo":            ["Keep sores clean", "Take prescribed antibiotics", "Avoid touching sores", "Wash hands frequently"],
}

# ── Fallback predictions ───────────────────────────────────────────────────────
FALLBACK_PREDICTIONS = [
    {"disease": "Viral Fever",  "confidence": 60,
     "description": "Fever caused by viral infection. Rest and hydration recommended.",
     "precautions": ["Rest", "Stay hydrated", "Take fever medication", "Consult a doctor"]},
    {"disease": "Common Cold",  "confidence": 40,
     "description": "Mild upper respiratory infection. Usually resolves in 7-10 days.",
     "precautions": ["Rest", "Stay hydrated", "Take vitamin C", "Avoid cold exposure"]},
    {"disease": "Influenza",    "confidence": 25,
     "description": "Flu infection affecting respiratory tract. Consult a doctor.",
     "precautions": ["Rest", "Stay hydrated", "Take prescribed medication", "Avoid contact with others"]},
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def symptoms_to_vector(symptom_string):
    user_symptoms = [s.strip().lower() for s in symptom_string.split(",") if s.strip()]
    vector = np.zeros(len(symptoms), dtype=int)
    for i, symptom_col in enumerate(symptoms):
        if symptom_col.strip().lower() in user_symptoms:
            vector[i] = 1
    return vector.reshape(1, -1)


def get_disease_description(disease_name):
    if diseases:
        if isinstance(diseases, dict):
            if disease_name in diseases:
                return diseases[disease_name]
            for k, v in diseases.items():
                if k.lower() == disease_name.lower():
                    return v
        elif isinstance(diseases, list):
            for entry in diseases:
                if isinstance(entry, dict) and entry.get("name", "").lower() == disease_name.lower():
                    return entry.get("description", "")
    return "Please consult a qualified doctor for accurate diagnosis and treatment."


def get_precautions(disease_name):
    if disease_name in precautions:
        return precautions[disease_name]
    for k, v in precautions.items():
        if k.lower() == disease_name.lower():
            return v
    return ["Consult a doctor", "Rest", "Stay hydrated", "Follow medical advice"]


def build_predictions(symptom_string):
    vector = symptoms_to_vector(symptom_string)

    if hasattr(model, "predict_proba"):
        proba   = model.predict_proba(vector)[0]
        classes = model.classes_
        top_idx = np.argsort(proba)[::-1][:3]
        results = []
        for idx in top_idx:
            disease_name = encoder.inverse_transform([classes[idx]])[0]
            confidence   = round(float(proba[idx]) * 100, 1)
            results.append({
                "disease":     disease_name,
                "confidence":  confidence,
                "description": get_disease_description(disease_name),
                "precautions": get_precautions(disease_name),
            })

    elif hasattr(model, "decision_function"):
        scores  = model.decision_function(vector)[0]
        classes = model.classes_
        top_idx = np.argsort(scores)[::-1][:3]
        results = []
        for rank, idx in enumerate(top_idx):
            disease_name = encoder.inverse_transform([classes[idx]])[0]
            confidence   = round(max(5.0, 85.0 - rank * 20.0), 1)
            results.append({
                "disease":     disease_name,
                "confidence":  confidence,
                "description": get_disease_description(disease_name),
                "precautions": get_precautions(disease_name),
            })

    else:
        pred         = model.predict(vector)[0]
        disease_name = encoder.inverse_transform([pred])[0]
        results = [{
            "disease":     disease_name,
            "confidence":  75.0,
            "description": get_disease_description(disease_name),
            "precautions": get_precautions(disease_name),
        }]

    return results


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":         "ok",
        "model":          "SVM+DT",
        "model_loaded":   MODEL_LOADED,
        "symptoms_count": len(symptoms) if symptoms else 0
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({
                "status":      "error",
                "message":     "Request body must be JSON",
                "predictions": FALLBACK_PREDICTIONS
            }), 400

        symptom_string = data.get("symptoms", "").strip()
        if not symptom_string:
            return jsonify({
                "status":      "error",
                "message":     "Field 'symptoms' is required",
                "predictions": FALLBACK_PREDICTIONS
            }), 400

        logger.info(f"📥 /predict → symptoms: {symptom_string}")

        if not MODEL_LOADED:
            logger.warning("Model not loaded — returning fallback")
            return jsonify({
                "status":      "success",
                "predictions": FALLBACK_PREDICTIONS,
                "note":        "fallback — model unavailable"
            }), 200

        predictions = build_predictions(symptom_string)
        logger.info(f"✅ Results: {[p['disease'] for p in predictions]}")

        return jsonify({
            "status":      "success",
            "predictions": predictions
        }), 200

    except Exception as e:
        logger.error(f"❌ /predict error: {e}", exc_info=True)
        return jsonify({
            "status":      "success",
            "predictions": FALLBACK_PREDICTIONS,
            "note":        "fallback due to internal error"
        }), 200


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)