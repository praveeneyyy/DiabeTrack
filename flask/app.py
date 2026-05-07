import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key
model = pickle.load(open('model.pkl', 'rb'))

dataset = pd.read_csv('diabetes.csv')

dataset_X = dataset.iloc[:,[1, 4, 5, 7]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start')
def start():
    return render_template('personal.html')

@app.route('/process_personal', methods=['POST'])
def process_personal():
    session['age'] = int(request.form['age'])
    session['gender'] = request.form['gender']
    session['height'] = float(request.form['height'])
    session['weight'] = float(request.form['weight'])
    # Calculate BMI
    height_m = session['height'] / 100
    session['bmi'] = round(session['weight'] / (height_m ** 2), 2)
    return redirect(url_for('question_1'))

@app.route('/question/1')
def question_1():
    return render_template('question1.html')

@app.route('/process_question_1', methods=['POST'])
def process_question_1():
    session['family_history'] = request.form['family_history']
    session['frequent_urination'] = request.form['frequent_urination']
    session['excessive_thirst'] = request.form['excessive_thirst']
    session['frequent_hunger'] = request.form['frequent_hunger']
    session['sudden_weight_loss'] = request.form['sudden_weight_loss']
    return redirect(url_for('question_2'))

@app.route('/question/2')
def question_2():
    return render_template('question2.html')

@app.route('/process_question_2', methods=['POST'])
def process_question_2():
    session['tired_weak'] = request.form['tired_weak']
    session['blurred_vision'] = request.form['blurred_vision']
    session['slow_healing'] = request.form['slow_healing']
    session['numbness_tingling'] = request.form['numbness_tingling']
    session['frequent_infections'] = request.form['frequent_infections']
    return redirect(url_for('question_3'))

@app.route('/question/3')
def question_3():
    return render_template('question3.html')

@app.route('/process_question_3', methods=['POST'])
def process_question_3():
    session['high_bp'] = request.form['high_bp']
    session['high_cholesterol'] = request.form['high_cholesterol']
    session['blood_glucose'] = request.form.get('blood_glucose', '')
    session['hba1c'] = request.form.get('hba1c', '')
    session['exercise'] = request.form['exercise']
    session['sugary_foods'] = request.form['sugary_foods']
    session['smoke'] = request.form['smoke']
    session['alcohol'] = request.form['alcohol']
    session['sleep_hours'] = float(request.form['sleep_hours'])
    session['stress'] = request.form['stress']
    # Gender-specific
    if session['gender'] == 'female':
        return redirect(url_for('question_female'))
    else:
        # For males, set defaults for female questions
        session['prediabetes'] = 'no'
        session['diabetes_medication'] = 'no'
        session['gestational_diabetes'] = 'n/a'
        session['pcos'] = 'n/a'
        return redirect(url_for('predict_report'))

@app.route('/question/female')
def question_female():
    return render_template('question_female.html')

@app.route('/process_female', methods=['POST'])
def process_female():
    session['prediabetes'] = request.form['prediabetes']
    session['diabetes_medication'] = request.form['diabetes_medication']
    session['gestational_diabetes'] = request.form['gestational_diabetes']
    session['pcos'] = request.form['pcos']
    return redirect(url_for('predict_report'))

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict( sc.transform(final_features) )

    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred

    return render_template('index.html', prediction_text='{}'.format(output))

@app.route('/predict_report')
def predict_report():
    # Get values for prediction
    blood_glucose = session.get('blood_glucose', '').strip()
    glucose = float(blood_glucose) if blood_glucose else 100.0  # default 100
    insulin = 79.8  # mean
    bmi = session['bmi']
    age = session['age']
    
    features = np.array([[glucose, insulin, bmi, age]])
    scaled_features = sc.transform(features)
    prediction = model.predict(scaled_features)[0]
    
    session['prediction'] = int(prediction)
    session['prediction_status'] = 'Positive' if prediction == 1 else 'Negative'
    session['prediction_text'] = (
        f"Diabetes {session['prediction_status']}. Please consult a doctor." 
        if prediction == 1 else 
        f"Diabetes {session['prediction_status']}. Keep monitoring your health."
    )
    
    return render_template('report.html')

if __name__ == "__main__":
    app.run(debug=True)
