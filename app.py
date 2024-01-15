from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the trained model
#model = tf.keras.models.load_model('emoji_sentiment_model.h5')

@app.route('/')
def index():
    return render_template('home.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get the drawn emoji data from the request
#         drawn_data = request.json['drawn_data']

#         # Preprocess the drawn data (replace this with your actual preprocessing steps)
#         # For example, you may need to resize the image, normalize pixel values, etc.
#         preprocessed_data = preprocess_drawn_data(drawn_data)

#         # Make a prediction using the loaded model
#         prediction = model.predict(preprocessed_data)

#         # Get the predicted sentiment label (replace this with your post-processing logic)
#         predicted_label = np.argmax(prediction)

#         # Map the label to a sentiment category (replace this with your label-to-sentiment mapping)
#         sentiments = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral']
#         predicted_sentiment = sentiments[predicted_label]

#         return jsonify({'predicted_sentiment': predicted_sentiment})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# def preprocess_drawn_data(drawn_data):
#     # Replace this with your actual preprocessing steps (resize, normalize, etc.)
#     # Convert drawn_data to a format suitable for your model
#     #def preprocess_drawn_data(drawn_data):
#     # Convert drawn_data to a NumPy array with appropriate data type
#     preprocessed_data = np.array([drawn_data], dtype=np.float32)
#     return preprocessed_data


if __name__ == '__main__':
    app.run(debug=True)
