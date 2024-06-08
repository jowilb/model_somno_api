import numpy as np
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model
with open('./model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        prediction = model.predict([np.array(data['features'])])
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/')
def index():
    return 'Hello, World! 123'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
