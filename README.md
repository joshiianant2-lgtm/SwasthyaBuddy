# 🩺 SwasthyaBuddy

An ML-powered web application that predicts possible diseases based on symptoms selected by the user and suggests basic precautions for early self-care.

![SwasthyaBuddy Demo](screenshots/demo.png)

## ✨ Features

- 🔍 Search through 133 symptoms instantly
- 🖱️ Click to select multiple symptoms
- 🔬 Predicts disease using a trained ML model
- 📊 Shows Top 5 possible diseases with confidence chart
- ⚠️ Provides precautions for the predicted disease
- 📱 Responsive and clean UI

## 🤖 Model

| Model | Accuracy |
|-------|----------|
| Random Forest | 97.62% |
| SVM | 100.00% |
| Gradient Boosting | 97.62% |

> Final model used: **Random Forest** (chosen for better probability scores)

- Dataset: [Disease Prediction Dataset](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)
- 41 diseases, 133 symptoms

## 🛠️ Tech Stack

- **Backend:** Python, Flask, scikit-learn
- **Frontend:** HTML, CSS, JavaScript, Chart.js
- **ML:** Random Forest Classifier, NumPy, Pandas

## 🚀 How to Run Locally

1. Clone the repository
```
git clone https://github.com/joshiianant2-lgtm/SwasthyaBuddy.git
cd SwasthyaBuddy
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Run the app
```
python app.py
```

4. Open browser at `http://127.0.0.1:5000`

## 📁 Project Structure
```
SwasthyaBuddy/
├── model/
│   ├── model.pkl          # Trained ML model
│   ├── encoder.pkl        # Label encoder
│   ├── symptoms.json      # List of symptoms
│   └── diseases.json      # List of diseases
├── templates/
│   └── index.html         # Frontend UI
├── static/
│   └── style.css          # Styling
├── app.py                 # Flask application
├── requirements.txt
└── README.md
```

## ⚕️ Disclaimer

This app is for educational purposes only and is not a substitute for professional medical advice. Always consult a qualified doctor.
```

