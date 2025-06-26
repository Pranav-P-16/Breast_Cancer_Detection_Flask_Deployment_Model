from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('breast_cancer_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            uploaded_file = request.files['file']
            if uploaded_file.filename.endswith('.txt'):
                content = uploaded_file.read().decode('utf-8')
                values = content.replace(',', ' ').split()
                if len(values) != 30:
                    result = 'File must contain exactly 30 numeric comma separated values.'
                else:
                    features = [float(v) for v in values]
                    input_array = np.array(features).reshape(1, -1)
                    prediction = model.predict(input_array)
                    result = 'Malignant / Cancerous' if prediction[0] == 0 else 'Benign / Non-cancerous'
            else:
                result = 'Please upload a valid .txt file.'
        except Exception as e:
            result = f'Error processing file: {str(e)}'
    return render_template('index.html', result=result)
if __name__ == "__main__":
    app.run(debug=True)