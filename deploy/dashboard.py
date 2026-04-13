import os
import time
import json
from flask import Flask, render_template_string, Response, jsonify, send_from_directory

app = Flask(__name__)

# Paths - match what C++ writes to
PREVIEW_PATH = "/dev/shm/preview.jpg"
STATUS_PATH = "/dev/shm/status.json"
ARTIFACTS_DIR = os.path.expanduser("~/Desktop/deploy/cpp_demo/artifacts")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Fight Detection Dashboard</title>
    <style>
        body { font-family: sans-serif; background: #1a1a1a; color: #fff; margin: 0; padding: 20px; }
        .container { display: flex; gap: 20px; }
        .video-box { flex: 2; background: #000; border-radius: 8px; overflow: hidden; position: relative; }
        .video-box img { width: 100%; height: auto; display: block; }
        .status-box { flex: 1; background: #333; padding: 20px; border-radius: 8px; }
        .status-indicator { 
            padding: 15px; text-align: center; font-size: 24px; font-weight: bold; 
            border-radius: 4px; margin-bottom: 20px;
        }
        .safe { background: #2ecc71; color: white; }
        .danger { background: #e74c3c; color: white; animation: blink 1s infinite; }
        @keyframes blink { 50% { opacity: 0.5; } }
        .info-row { display: flex; justify-content: space-between; margin-bottom: 10px; border-bottom: 1px solid #444; padding-bottom: 5px; }
        .clip-list { height: 300px; overflow-y: auto; }
        .clip-item { 
            padding: 10px; border-bottom: 1px solid #444; cursor: pointer; display: flex; justify-content: space-between;
        }
        .clip-item:hover { background: #444; }
        a { color: #3498db; text-decoration: none; }
    </style>
</head>
<body>
    <h1>🛡️ AI Violence Detection System</h1>
    <div class="container">
        <div class="video-box">
            <img src="/video_feed" alt="Live Feed">
        </div>
        <div class="status-box">
            <div id="status-indicator" class="status-indicator safe">SAFE</div>
            
            <div class="info-row">
                <span>Probability:</span>
                <span id="prob-val">0.00</span>
            </div>
            <div class="info-row">
                <span>Label:</span>
                <span id="label-val">--</span>
            </div>

            <h3>🚨 Incident Clips</h3>
            <div class="clip-list" id="clip-list">
                <!-- Clips will be loaded here -->
            </div>
        </div>
    </div>

    <script>
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    const ind = document.getElementById('status-indicator');
                    document.getElementById('prob-val').innerText = parseFloat(data.prob).toFixed(4);
                    document.getElementById('label-val').innerText = data.label;
                    
                    if (data.isFight === "true") {
                        ind.className = 'status-indicator danger';
                        ind.innerText = 'VIOLENCE DETECTED';
                    } else {
                        ind.className = 'status-indicator safe';
                        ind.innerText = 'SAFE';
                    }
                });
        }
        
        function updateClips() {
            fetch('/clips')
                .then(response => response.json())
                .then(data => {
                    const list = document.getElementById('clip-list');
                    list.innerHTML = '';
                    data.forEach(clip => {
                        const div = document.createElement('div');
                        div.className = 'clip-item';
                        div.innerHTML = '<span>' + clip + '</span> <a href="/download/' + clip + '" target="_blank">View</a>';
                        list.appendChild(div);
                    });
                });
        }

        setInterval(updateStatus, 500);
        setInterval(updateClips, 5000);
        updateClips();
    </script>
</body>
</html>
"""

def generate_frames():
    while True:
        if os.path.exists(PREVIEW_PATH):
            try:
                with open(PREVIEW_PATH, "rb") as f:
                    data = f.read()
                    if data:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
            except:
                pass
        time.sleep(0.1)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    if os.path.exists(STATUS_PATH):
        try:
            with open(STATUS_PATH, "r") as f:
                return jsonify(json.load(f))
        except:
            pass
    return jsonify({"isFight": "false", "prob": 0, "label": "offline"})

@app.route('/clips')
def list_clips():
    clips = []
    if os.path.exists(ARTIFACTS_DIR):
        clips = [f for f in os.listdir(ARTIFACTS_DIR) if f.startswith("event_") and f.endswith(".mp4")]
        clips.sort(reverse=True) # Newest first
    return jsonify(clips)

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(ARTIFACTS_DIR, filename)

if __name__ == '__main__':
    # Ensure artifacts dir exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, threaded=True)
