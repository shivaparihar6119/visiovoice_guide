from flask import Flask, render_template, Response
import time
import cv2 as cv
import numpy as np
import pyttsx3
from threading import Thread

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

known_dist = 45
known_width = 14.3
person_width = 16.0  # inches
mobile_width = 3  # inches

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3
fonts = cv.FONT_HERSHEY_COMPLEX
cap = cv.VideoCapture(0)
# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

class_names = []

with open("../classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]

        label = "%s : %f" % (class_names[int(classid)], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1] - 14), fonts, 0.5, color, 2)

        if classid == 0:  # person class id
            data_list.append([class_names[int(classid)], box[2], (box[0], box[1] - 2)])
        elif classid == 67:
            data_list.append([class_names[int(classid)], box[2], (box[0], box[1] - 2)])

    return data_list

def focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value

def distance_measure(focal_length, real_face_width, face_width_in_frame):
    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance

ref_person = cv.imread('../image14.png')
ref_mobile = cv.imread('../image4.png')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")
focal_person = focal_length(known_dist, person_width, person_width_in_rf)

focal_mobile = focal_length(known_dist, mobile_width, mobile_width_in_rf)

last_check_time = time.time()

app = Flask(__name__)

def play_audio(text):
    engine.say(text)
    engine.runAndWait()

def generate_frames():
    while True:
        ret, frame = cap.read()
        data = object_detector(frame)
        frame_width = frame.shape[1]
        obj_centre = 0

        text = ""
        for d in data:
            if d[0] == 'person':
                distance = distance_measure(focal_person, person_width, d[1])
                x, y = d[2]
                obj_centre = x + d[1] / 2
                text += analyze_position(obj_centre, frame_width, distance, d[0])
            elif d[0] == 'cell phone':
                distance = distance_measure(focal_mobile, mobile_width, d[1])
                x, y = d[2]
                obj_centre = x + d[1] / 2
                text += analyze_position(obj_centre, frame_width, distance, d[0])

            cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
            cv.putText(frame, f'Dis: {round(distance, 2)} inch', (x + 5, y + 13), fonts, 0.48, GREEN, 2)

        if time.time() - last_check_time >= 3:
            audio_thread = Thread(target=play_audio, args=(text,))
            audio_thread.start()
            last_check_time = time.time()

        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def analyze_position(obj_centre, frame_width, distance, obj_type):
    if obj_centre < frame_width / 5:
        return f'The {obj_type} is at left at {round(distance, 2)} inches\n'
    elif obj_centre < 2 * (frame_width / 5):
        return f'The {obj_type} is at slightly left at {round(distance, 2)} inches\n'
    elif obj_centre < 3 * (frame_width / 5):
        return f'The {obj_type} is at centre at{round(distance, 2)} inches\n'
    elif obj_centre < 4 * (frame_width / 5):
        return f'The {obj_type} is at slightly right at {round(distance, 2)} inches\n'
    else:
        return f'The {obj_type} is at right at{round(distance, 2)} inches\n'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)