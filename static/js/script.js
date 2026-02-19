// Global variables
let videoElement = document.getElementById('webcam');
let canvasElement = document.getElementById('output-canvas');
let canvasCtx = canvasElement.getContext('2d');
let gestureList = document.getElementById('gesture-list');
let systemStatus = document.getElementById('system-status');
let predictionSpan = document.getElementById('gesture-name');
let recognitionMode = document.getElementById('recognition-mode');
let retrainBtn = document.getElementById('retrain-btn');
let startRecordBtn = document.getElementById('start-record');
let recordStatus = document.getElementById('record-status');

let currentGestures = [];
let recording = false;
let recordingGestureId = null;
let recordingInterval = null;

// Initialize MediaPipe Hands
const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

hands.onResults(onHandResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({image: videoElement});
    },
    width: 640,
    height: 480
});
camera.start();

// Helper: flatten landmarks to array of 63 numbers
function landmarksToArray(landmarks) {
    let arr = [];
    for (let lm of landmarks) {
        arr.push(lm.x, lm.y, lm.z);
    }
    return arr;
}

// Heuristic MediaPipe classifier (simple rules for demo)
function heuristicGesture(landmarks) {
    // Very basic: thumb up vs peace vs fist
    // This is just placeholder; you can expand with real logic
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const middleTip = landmarks[12];
    const ringTip = landmarks[16];
    const pinkyTip = landmarks[20];

    // Thumb up: thumb above other fingers (simplified)
    if (thumbTip.y < indexTip.y - 0.1 && thumbTip.y < middleTip.y - 0.1) {
        return "thumb_up";
    }
    // Peace: index and middle extended, others bent
    if (indexTip.y < ringTip.y && middleTip.y < ringTip.y && 
        Math.abs(indexTip.x - middleTip.x) > 0.05) {
        return "peace";
    }
    // Fist: all fingers below middle of palm (simplified)
    const palmY = landmarks[0].y;
    if (indexTip.y > palmY + 0.1 && middleTip.y > palmY + 0.1) {
        return "fist";
    }
    return "unknown";
}

// Process hand results
function onHandResults(results) {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        // Draw landmarks
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 2});
            drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', radius: 3});
        }

        // Get landmarks of first hand
        const landmarks = results.multiHandLandmarks[0];
        const lmArray = landmarksToArray(landmarks);

        // Determine mode and predict
        const mode = recognitionMode.value;
        if (mode === 'mediapipe') {
            const gesture = heuristicGesture(landmarks);
            predictionSpan.innerText = gesture;
            // Execute action if mapped (we need to get action from currentGestures)
            const gestureObj = currentGestures.find(g => g.name === gesture);
            if (gestureObj && gestureObj.action) {
                fetch('/api/execute', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action: gestureObj.action})
                });
            }
        } else if (mode === 'cnn') {
            // Send landmarks to backend for CNN prediction
            fetch('/api/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({landmarks: lmArray})
            })
            .then(res => res.json())
            .then(data => {
                if (data.gesture) {
                    predictionSpan.innerText = data.gesture;
                    if (data.action) {
                        fetch('/api/execute', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({action: data.action})
                        });
                    }
                } else {
                    predictionSpan.innerText = 'None';
                }
            });
        }

        // If recording, save this sample
        if (recording && recordingGestureId) {
            fetch('/api/samples', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    gesture_id: recordingGestureId,
                    landmarks: lmArray
                })
            });
        }
    } else {
        predictionSpan.innerText = 'No hand';
    }
}

// Load gestures from backend and render
function loadGestures() {
    fetch('/api/gestures')
        .then(res => res.json())
        .then(data => {
            currentGestures = data;
            renderGestures();
        });
}

function renderGestures() {
    gestureList.innerHTML = '';
    currentGestures.forEach(g => {
        const div = document.createElement('div');
        div.className = 'gesture-item';
        div.innerHTML = `
            <span><strong>${g.name}</strong></span>
            <select data-id="${g.id}" class="action-select">
                <option value="play_pause" ${g.action === 'play_pause' ? 'selected' : ''}>Play/Pause</option>
                <option value="next_track" ${g.action === 'next_track' ? 'selected' : ''}>Next Track</option>
                <option value="prev_track" ${g.action === 'prev_track' ? 'selected' : ''}>Previous Track</option>
                <option value="volume_up" ${g.action === 'volume_up' ? 'selected' : ''}>Volume Up</option>
                <option value="volume_down" ${g.action === 'volume_down' ? 'selected' : ''}>Volume Down</option>
                <option value="mute" ${g.action === 'mute' ? 'selected' : ''}>Mute</option>
                <option value="tab_next" ${g.action === 'tab_next' ? 'selected' : ''}>Next Tab</option>
                <option value="tab_prev" ${g.action === 'tab_prev' ? 'selected' : ''}>Previous Tab</option>
                <option value="window_next" ${g.action === 'window_next' ? 'selected' : ''}>Next Window</option>
                <option value="window_prev" ${g.action === 'window_prev' ? 'selected' : ''}>Previous Window</option>
                <option value="browser_back" ${g.action === 'browser_back' ? 'selected' : ''}>Browser Back</option>
                <option value="browser_forward" ${g.action === 'browser_forward' ? 'selected' : ''}>Browser Forward</option>
            </select>
            <button class="delete-gesture" data-id="${g.id}">Delete</button>
        `;
        gestureList.appendChild(div);
    });

    // Attach event listeners
    document.querySelectorAll('.action-select').forEach(select => {
        select.addEventListener('change', (e) => {
            const gid = e.target.dataset.id;
            const action = e.target.value;
            fetch(`/api/gestures/${gid}/action`, {
                method: 'PUT',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action})
            });
        });
    });

    document.querySelectorAll('.delete-gesture').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const gid = e.target.dataset.id;
            fetch(`/api/gestures/${gid}`, {method: 'DELETE'})
                .then(() => loadGestures());
        });
    });
}

// Add new gesture
document.getElementById('start-record').addEventListener('click', () => {
    const name = document.getElementById('new-gesture-name').value.trim();
    const action = document.getElementById('new-gesture-action').value;
    if (!name) {
        alert('Please enter a gesture name');
        return;
    }

    // First create gesture
    fetch('/api/gestures', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name, action})
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }
        // Start recording samples for this gesture
        recordingGestureId = data.id;
        recording = true;
        recordStatus.innerText = 'Recording... Hold gesture steady';
        systemStatus.innerText = 'Recording';
        systemStatus.style.background = '#f44336';

        // Stop after 5 seconds
        setTimeout(() => {
            recording = false;
            recordingGestureId = null;
            recordStatus.innerText = '';
            systemStatus.innerText = 'System Ready';
            systemStatus.style.background = '#4CAF50';
            loadGestures();
        }, 5000);
    });
});

// Retrain button
retrainBtn.addEventListener('click', () => {
    systemStatus.innerText = 'Training...';
    systemStatus.style.background = '#FF9800';
    fetch('/api/retrain', {method: 'POST'})
        .then(res => res.json())
        .then(data => {
            systemStatus.innerText = 'Model trained';
            systemStatus.style.background = '#4CAF50';
            setTimeout(() => {
                systemStatus.innerText = 'System Ready';
            }, 2000);
        });
});

// Initial load
loadGestures();