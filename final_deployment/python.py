from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import os
import mediapipe as mp
import numpy as np
import io
import torch
from collections import deque
from RNN_model import RNN

# -----------------------------------------------------------
# Flask App Initialization
# -----------------------------------------------------------
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Allow all origins for simplicity

# -----------------------------------------------------------
# Model and Pose Setup
# -----------------------------------------------------------
# Model path (ensure this matches your structure)
model_path = os.path.join(os.path.dirname(__file__), 'rnn_epoch73_loss0.19.pth')

# Load the trained model once at startup
model = RNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Frame buffer for temporal sequence
frame_buffer = deque(maxlen=40)

# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------

# Serve the frontend page
@app.route('/')
def home():
    # Serves front.html directly
    return send_from_directory('.', 'front.html')

# -----------------------------------------------------------
# Helper Function
# -----------------------------------------------------------
def extract_skeleton(frame):
    """Extract pose skeleton from an image frame."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        skeleton = []
        for lm in landmarks:
            skeleton.extend([lm.x, lm.y, lm.z])
        return skeleton
    else:
        # Return zero array if no landmarks found
        return [0.0] * 33 * 3

# -----------------------------------------------------------
# Prediction Endpoint
# -----------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame received'})

    file = request.files['frame']
    in_memory = io.BytesIO()
    file.save(in_memory)

    # Convert to numpy image
    np_img = np.frombuffer(in_memory.getvalue(), np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Extract skeleton
    skeleton = extract_skeleton(frame)
    frame_buffer.append(skeleton)

    # When 40 frames collected → predict
    if len(frame_buffer) == 40:
        input_tensor = torch.tensor(list(frame_buffer), dtype=torch.float32)

        # Check if all frames are blank
        if torch.all(input_tensor.eq(0)):
            frame_buffer.clear()
            return jsonify({'result': 'No gesture detected'})

        # Run model
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 0)
            result = int(predicted.item())

        frame_buffer.clear()
        return jsonify({'result': result})

    return jsonify({'result': 'Frame received'})

# -----------------------------------------------------------
# Start Server
# -----------------------------------------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
