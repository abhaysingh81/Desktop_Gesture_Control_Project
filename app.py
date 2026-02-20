from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import gesture_db as db
import desktop_control as dc
import model as ml
import torch
import numpy as np
from model import EmbeddingNet, GestureDB, train_embedding_model, MODEL_PATH, DEVICE
import gesture_db as db   
import os
import numpy as np

app = Flask(__name__)
CORS(app)
db.init_db()

# Global state
current_model = None
gesture_labels = []  # list of gesture names in order of classes
embedding_net = EmbeddingNet().to(DEVICE)
if os.path.exists(MODEL_PATH):
    embedding_net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
gesture_db = GestureDB(model=embedding_net)


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
    # Get name before deleting
    gestures = db.get_gestures()
    gesture_name = next((g['name'] for g in gestures if g['id'] == gid), None)
    db.delete_gesture(gid)
    if gesture_name:
        gesture_db.remove_gesture(gesture_name)
    return jsonify({'success': True})

@app.route('/api/gestures/<int:gid>/action', methods=['PUT'])
def update_action(gid):
    data = request.json
    action = data.get('action')
    db.update_gesture_action(gid, action)
    # Also update in gesture_db
    gestures = db.get_gestures()
    gesture_name = next((g['name'] for g in gestures if g['id'] == gid), None)
    if gesture_name:
        gesture_db.update_action(gesture_name, action)
    return jsonify({'success': True})

@app.route('/api/samples', methods=['POST'])
def add_sample():
    data = request.json
    gid = data.get('gesture_id')
    landmarks = data.get('landmarks')
    if not gid or not landmarks:
        return jsonify({'error': 'Missing data'}), 400
    # 1. Store in SQLite (for future retraining)
    db.add_sample(gid, landmarks)
    # 2. Update GestureDB (need gesture name)
    gestures = db.get_gestures()
    gesture_name = next((g['name'] for g in gestures if g['id'] == gid), None)
    if gesture_name:
        # We add this single sample to the live index
        gesture_db.add_gesture_samples(gesture_name, "", [landmarks])  # action not updated here
    return jsonify({'success': True})

@app.route('/api/retrain', methods=['POST'])
def retrain():
    # 1. Collect all samples from SQLite
    samples_data = db.get_all_samples()   # returns list of (gesture_name, landmarks)
    if not samples_data:
        return jsonify({'error': 'No samples'}), 400
    # Prepare lists for training
    gestures = db.get_gestures()
    name_to_action = {g['name']: g['action'] for g in gestures}
    # We need integer labels for training – map names to indices
    unique_names = list(set([name for name, _ in samples_data]))
    name_to_idx = {name: i for i, name in enumerate(unique_names)}
    landmarks_list = []
    labels_list = []
    for name, lm in samples_data:
        landmarks_list.append(lm)
        labels_list.append(name_to_idx[name])

    # 2. Train new embedding model
    model = train_embedding_model(landmarks_list, labels_list)
    if model is None:
        return jsonify({'error': 'Training failed (need at least 2 classes)'}), 400

    # 3. Update global model in gesture_db
    gesture_db.model = model
    # 4. Rebuild gesture_db embeddings from all samples (using new model)
    #    We also need to restore actions
    gesture_db.rebuild_from_samples(samples_data)
    # 5. Update class_to_action from current gestures
    for g in gestures:
        gesture_db.class_to_action[g['name']] = g['action']
    gesture_db.save()

    return jsonify({'success': True, 'num_classes': len(unique_names)})

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    landmarks = data.get('landmarks')
    if not landmarks:
        return jsonify({'error': 'No landmarks'}), 400
    gesture, action = gesture_db.predict(landmarks, k=3)  
    return jsonify({'gesture': gesture, 'action': action})

@app.route('/api/execute', methods=['POST'])
def execute():
    data = request.json
    action = data.get('action')
    if action:
        dc.execute_action(action)
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)