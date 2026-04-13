import cv2
import time
import threading
from flask import Flask, Response

app = Flask(__name__)
video_path = "/home/HwHiAiUser/Desktop/deploy/fn.mp4"

def generate_frames():
    cap = cv2.VideoCapture(video_path)
    while True:
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                time.sleep(1)
                continue
        
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Limit FPS to simulate real camera (~25 FPS)
        time.sleep(0.04)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run on port 5555 as expected by the C++ demo
    print("Starting video stream on port 5555...")
    app.run(host='0.0.0.0', port=5555, debug=False, threaded=True)
