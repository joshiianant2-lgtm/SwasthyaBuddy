from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np

app = Flask(__name__)

# Load model and files
model = pickle.load(open('model/model.pkl', 'rb'))
encoder = pickle.load(open('model/encoder.pkl', 'rb'))
symptoms = sorted(json.load(open('model/symptoms.json')))
diseases = json.load(open('model/diseases.json'))

# Basic precautions for each disease
precautions = {
    "Fungal infection": ["Keep skin clean and dry", "Use antifungal cream", "Avoid sharing personal items", "Wear breathable clothing"],
    "Allergy": ["Avoid allergens", "Take antihistamines", "Consult a doctor", "Keep windows closed during high pollen"],
    "GERD": ["Avoid spicy food", "Eat smaller meals", "Don't lie down after eating", "Avoid caffeine and alcohol"],
    "Chronic cholestasis": ["Avoid alcohol", "Follow low-fat diet", "Take prescribed medication", "Regular liver checkups"],
    "Drug Reaction": ["Stop the medication immediately", "Consult a doctor", "Drink plenty of water", "Avoid self-medication"],
    "Peptic ulcer diseae": ["Avoid spicy food", "Avoid alcohol", "Take antacids", "Eat smaller frequent meals"],
    "AIDS": ["Use protection", "Regular medical checkups", "Take antiretroviral drugs", "Maintain healthy lifestyle"],
    "Diabetes": ["Monitor blood sugar", "Follow diabetic diet", "Exercise regularly", "Take prescribed medication"],
    "Gastroenteritis": ["Drink plenty of fluids", "Rest", "Avoid solid food initially", "Wash hands frequently"],
    "Bronchial Asthma": ["Avoid triggers", "Use inhaler as prescribed", "Stay indoors on high pollution days", "Practice breathing exercises"],
    "Hypertension": ["Reduce salt intake", "Exercise regularly", "Avoid stress", "Monitor blood pressure daily"],
    "Migraine": ["Rest in dark quiet room", "Stay hydrated", "Avoid triggers like bright light", "Take prescribed medication"],
    "Cervical spondylosis": ["Do neck exercises", "Use ergonomic pillow", "Avoid straining neck", "Apply heat/cold pack"],
    "Paralysis (brain hemorrhage)": ["Seek emergency care immediately", "Follow rehabilitation plan", "Take prescribed medication", "Regular physiotherapy"],
    "Jaundice": ["Drink plenty of water", "Rest", "Avoid alcohol", "Follow doctor's diet plan"],
    "Malaria": ["Take antimalarial medication", "Use mosquito nets", "Apply insect repellent", "Avoid stagnant water areas"],
    "Chicken pox": ["Avoid scratching", "Take antihistamines", "Stay isolated", "Keep skin clean"],
    "Dengue": ["Rest and stay hydrated", "Take paracetamol only", "Avoid aspirin", "Use mosquito repellent"],
    "Typhoid": ["Drink boiled water", "Eat freshly cooked food", "Take prescribed antibiotics", "Rest"],
    "Hepatitis A": ["Rest", "Drink plenty of fluids", "Avoid alcohol", "Eat light nutritious food"],
    "Hepatitis B": ["Get vaccinated", "Avoid alcohol", "Use protection", "Regular liver monitoring"],
    "Hepatitis C": ["Avoid alcohol", "Take antiviral medication", "Regular checkups", "Avoid sharing needles"],
    "Hepatitis D": ["Get Hepatitis B vaccine", "Avoid alcohol", "Take prescribed medication", "Regular monitoring"],
    "Hepatitis E": ["Drink clean water", "Eat cooked food", "Rest", "Avoid alcohol"],
    "Alcoholic hepatitis": ["Stop alcohol immediately", "Follow prescribed diet", "Regular liver checkups", "Join support group"],
    "Tuberculosis": ["Complete full course of antibiotics", "Cover mouth while coughing", "Get plenty of rest", "Eat nutritious food"],
    "Common Cold": ["Rest", "Drink warm fluids", "Take steam inhalation", "Avoid cold drinks"],
    "Pneumonia": ["Take prescribed antibiotics", "Rest", "Stay hydrated", "Follow up with doctor"],
    "Dimorphic hemmorhoids(piles)": ["Eat high fiber diet", "Drink plenty of water", "Avoid straining", "Sitz bath"],
    "Heart attack": ["Call emergency services immediately", "Chew aspirin if not allergic", "Rest", "Do not drive yourself"],
    "Varicose veins": ["Elevate legs", "Exercise regularly", "Avoid prolonged standing", "Wear compression stockings"],
    "Hypothyroidism": ["Take prescribed medication", "Regular thyroid checkups", "Exercise", "Eat iodine-rich food"],
    "Hyperthyroidism": ["Take prescribed medication", "Avoid caffeine", "Regular checkups", "Manage stress"],
    "Hypoglycemia": ["Eat small frequent meals", "Carry glucose tablets", "Avoid skipping meals", "Monitor blood sugar"],
    "Osteoarthritis": ["Exercise regularly", "Maintain healthy weight", "Use hot/cold packs", "Take prescribed medication"],
    "Arthritis": ["Exercise gently", "Apply warm compress", "Take anti-inflammatory medication", "Rest affected joints"],
    "(vertigo) Paroymsal  Positional Vertigo": ["Avoid sudden movements", "Do Epley maneuver", "Sleep with head elevated", "Consult ENT specialist"],
    "Acne": ["Keep face clean", "Avoid touching face", "Use non-comedogenic products", "Stay hydrated"],
    "Urinary tract infection": ["Drink plenty of water", "Take prescribed antibiotics", "Avoid holding urine", "Maintain hygiene"],
    "Psoriasis": ["Moisturize regularly", "Avoid triggers", "Use prescribed creams", "Manage stress"],
    "Impetigo": ["Keep affected area clean", "Take prescribed antibiotics", "Avoid touching sores", "Wash hands frequently"],
}

@app.route('/')
def index():
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    selected_symptoms = data.get('symptoms', [])
    
    # Build input vector
    input_vector = np.zeros(len(symptoms))
    for s in selected_symptoms:
        if s in symptoms:
            input_vector[symptoms.index(s)] = 1
    
    # Predict
    # Predict
    prediction = model.predict([input_vector])[0]
    disease = encoder.inverse_transform([prediction])[0]
    probs = model.predict_proba([input_vector])[0]
    confidence = round(float(np.max(probs)) * 100, 2)

    # Top 5 diseases with probabilities
    top5_indices = np.argsort(probs)[::-1][:5]
    top5 = [
        {
            'disease': encoder.inverse_transform([i])[0],
            'probability': round(float(probs[i]) * 100, 2)
        }
        for i in top5_indices
    ]

    # Get precautions
    disease_precautions = precautions.get(disease, ["Consult a doctor immediately"])

    return jsonify({
        'disease': disease,
        'confidence': confidence,
        'precautions': disease_precautions,
        'top5': top5
    })

if __name__ == '__main__':
    app.run(debug=True)