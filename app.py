from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

last_prediction = None  # store last result globally

@app.route('/')
def home():
    return render_template('index.html', last_prediction=last_prediction)

@app.route('/predict', methods=['POST'])
def predict():
    global last_prediction
    job_desc = request.form.get('job_description', '').strip()

    # Validation moved to Task 2 below (we’ll add it soon)
    if not job_desc:
        return render_template('index.html', error="Please enter a job description!")

    input_features = vectorizer.transform([job_desc])
    prediction = model.predict(input_features)[0]
    confidence = model.predict_proba(input_features)[0].max()

    last_prediction = {
        'prediction': prediction,
        'confidence': round(confidence * 100, 2)
    }

    return render_template('result.html', prediction=prediction, confidence=confidence)
    

if __name__ == '__main__':
    app.run(debug=True)

