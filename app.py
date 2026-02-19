from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import gesture_db as db
import desktop_control as dc
import model as ml
import torch
import numpy as np

app = Flask(__name__)
CORS(app)
db.init_db()

# Global state
current_model = None
gesture_labels = []  # list of gesture names in order of classes

def update_model():
    global current_model, gesture_labels
    samples_data = db.get_all_samples()
    if not samples_data:
        current_model = None
        gesture_labels = []
        return
    # Build samples and labels
    gestures = db.get_gestures()
    name_to_id = {g['name']: g['id'] for g in gestures}
    # We need numeric labels 0..N-1 consistent with gesture order
    gesture_names = [g['name'] for g in gestures]
    label_map = {name: i for i, name in enumerate(gesture_names)}
    X = []
    y = []
    for name, lm in samples_data:
        X.append(lm)
        y.append(label_map[name])
    # Train model
    model = ml.train_model(X, y)
    if model:
        current_model = model
        gesture_labels = gesture_names
    else:
        current_model = None
        gesture_labels = []

# ------- Routes --------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/gestures', methods=['GET'])
def get_gestures():
    return jsonify(db.get_gestures())

@app.route('/api/gestures', methods=['POST'])
def add_gesture():
    data = request.json
    name = data.get('name')
    action = data.get('action')
    if not name:
        return jsonify({'error': 'Name required'}), 400
    gid = db.add_gesture(name, action)
    if gid is None:
        return jsonify({'error': 'Gesture already exists'}), 400
    return jsonify({'id': gid, 'name': name, 'action': action})

@app.route('/api/gestures/<int:gid>', methods=['DELETE'])
def delete_gesture(gid):
    db.delete_gesture(gid)
    update_model()  # retrain after deletion? optional, but we have explicit retrain button
    return jsonify({'success': True})

@app.route('/api/gestures/<int:gid>/action', methods=['PUT'])
def update_action(gid):
    data = request.json
    action = data.get('action')
    db.update_gesture_action(gid, action)
    return jsonify({'success': True})

@app.route('/api/samples', methods=['POST'])
def add_sample():
    data = request.json
    gid = data.get('gesture_id')
    landmarks = data.get('landmarks')
    if not gid or not landmarks:
        return jsonify({'error': 'Missing data'}), 400
    db.add_sample(gid, landmarks)
    return jsonify({'success': True})

@app.route('/api/retrain', methods=['POST'])
def retrain():
    update_model()
    return jsonify({'success': True, 'num_gestures': len(gesture_labels)})

@app.route('/api/predict', methods=['POST'])
def predict():
    global current_model, gesture_labels
    data = request.json
    landmarks = data.get('landmarks')
    if not landmarks:
        return jsonify({'error': 'No landmarks'}), 400
    if current_model is None or not gesture_labels:
        return jsonify({'gesture': None, 'error': 'Model not trained'}), 400
    pred = ml.predict(landmarks, current_model, gesture_labels)
    # Find action for this gesture
    gestures = db.get_gestures()
    action = next((g['action'] for g in gestures if g['name'] == pred), None)
    return jsonify({'gesture': pred, 'action': action})

@app.route('/api/execute', methods=['POST'])
def execute():
    data = request.json
    action = data.get('action')
    if action:
        dc.execute_action(action)
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)