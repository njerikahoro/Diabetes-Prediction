from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('kisukari_model.joblib')
scaler = joblib.load('scaler2.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = [float(x) for x in request.form.values()]
        
        # Convert to numpy array and reshape
        input_data = np.array(data).reshape(1, -1)
        
        # Standardize the input data
        std_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(std_data)
        
        # Get prediction probability
        probability = model.predict_proba(std_data)
        
        # Prepare result
        if prediction[0] == 1:
            result = "Diabetic"
            confidence = probability[0][1]
        else:
            result = "Not Diabetic"
            confidence = probability[0][0]
        
        return render_template('index.html', 
                               prediction_text='The person is {} ({}% confidence)'.format(
                                   result, round(confidence * 100, 2)))
    
    except Exception as e:
        return render_template('index.html', 
                               prediction_text='Error in prediction: {}'.format(str(e)))

if __name__ == '__main__':
    app.run(debug=True)