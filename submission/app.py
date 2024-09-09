from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)
model = load_model("Signlang_translator.h5")  

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        print("Success:", success)
        print("Frame shape:", frame.shape)
        if not success:
            break
        else:
            frame_preprocess = preprocess_frame(frame)

            prediction = model.predict(frame_preprocess)

            frame = draw_prediction(frame, prediction)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def preprocess_frame(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frame_resized = cv2.resize(frame_gray, (28, 28))
    
    frame_normalized = frame_resized / 255.0
    
    frame_processed = np.expand_dims(frame_normalized, axis=0)
    
    return frame_processed


# def draw_prediction(frame, prediction):
#     predicted_class = np.argmax(prediction)
#     cv2.putText(frame, f"Predicted class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#     return frame

def draw_prediction(frame, prediction):
    # Map numerical class labels to characters
    class_to_char = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
        4: 'E',
        5: 'F',
        6: 'G',
        7: 'H',
        8: 'I',
        9: 'J',
        10: 'K',
        11: 'L',
        12: 'M',
        13: 'N',
        14: 'O',
        15: 'P',
        16: 'Q',
        17: 'R',
        18: 'S',
        19: 'T',
        20: 'U',
        21: 'V',
        22: 'W',
        23: 'X',
        24: 'Y',
        25: 'Z'
    }
    predicted_class = np.argmax(prediction)

    predicted_char = class_to_char.get(predicted_class, 'Unknown')

    cv2.putText(frame, f"Predicted sign: {predicted_char}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/convert', methods=["GET", "POST"])
def convert():
    if request.method == "GET":
        return render_template("convert.html")
    elif request.method == "POST":
        name = request.form['name'].upper()
        return render_template("text_to_sign.html", name = name)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
