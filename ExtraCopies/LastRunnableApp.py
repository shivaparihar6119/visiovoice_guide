from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import time
import cv2 as cv
import pyttsx3
from threading import Thread
from yolo import object_detector

app = Flask(__name__)


def create_app():
    return app


socketio = SocketIO(app)
cap = cv.VideoCapture(0)
last_check_time = time.time()
global_text = ""

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

known_dist = 45
known_width = 14.3
person_width = 16.0  # inches
mobile_width = 3  # inches
book_width = 16
ball_width = 2.8
bottle_width = 2.5


def focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value


def distance_measure(focal_length, real_face_width, face_width_in_frame):
    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance


ref_person = cv.imread('image14.png')
ref_mobile = cv.imread('image4.png')
ref_book = cv.imread('image5.png')
ref_ball = cv.imread('image2.png')
ref_bottle = cv.imread('image0.png')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

book_data = object_detector(ref_book)
book_width_in_rf = book_data[0][1]

ball_data = object_detector(ref_ball)
ball_width_in_rf = ball_data[0][1]

bottle_data = object_detector(ref_bottle)
bottle_width_in_rf = bottle_data[0][1]

print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")

focal_person = focal_length(known_dist, person_width, person_width_in_rf)
focal_mobile = focal_length(known_dist, mobile_width, mobile_width_in_rf)
focal_book = focal_length(known_dist, book_width, book_width_in_rf)
focal_ball = focal_length(known_dist, ball_width, ball_width_in_rf)
focal_bottle = focal_length(known_dist, bottle_width, bottle_width_in_rf)


def generate_frames():
    global last_check_time
    global global_text
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            data = object_detector(frame)
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]
            obj_centre = 0
            # def play_audio(text):
            #     engine.say(text)
            #     engine.runAndWait()

            text = ""
            for d in data:
                x, y = d[2]

                if d[0] == 'person':
                    distance = distance_measure(focal_person, person_width, d[1])
                elif d[0] == 'cell phone':
                    distance = distance_measure(focal_mobile, mobile_width, d[1])
                elif d[0] == 'book':
                    distance = distance_measure(focal_book, book_width, d[1])
                elif d[0] == 'sports ball':
                    distance = distance_measure(focal_ball, ball_width, d[1])
                elif d[0] == 'bottle':
                    distance = distance_measure(focal_bottle, bottle_width, d[1])

                x, y = d[2]
                obj_centre = x + d[1] / 2
                obj_vertical_center = y + d[1] / 2

                position_text = ""

                # Horizontal position
                if obj_centre < frame_width / 5:
                    position_text += 'left'
                elif obj_centre < 2 * (frame_width / 5):
                    position_text += 'slightly left'
                elif obj_centre < 3 * (frame_width / 5):
                    position_text += 'centre middle'
                elif obj_centre < 4 * (frame_width / 5):
                    position_text += 'slightly right'
                else:
                    position_text += 'right'

                # Vertical position
                if obj_vertical_center < frame_height / 3:
                    position_text += ' top'
                elif obj_vertical_center < 2 * (frame_height / 3):
                    position_text += ' center vertically'
                else:
                    position_text += ' bottom'

                # Combine both positions in the same sentence
                text += f'The {d[0]} is at {position_text} at {round(distance, 2)} inches\n'

                # obj_centre = x + d[1] / 2
                # if obj_centre < frame_width / 5:
                #     text += f'The {d[0]} is at left at {round(distance, 2)} inches\n'
                # elif obj_centre < 2 * (frame_width / 5):
                #     text += f'The {d[0]} is at slightly left at {round(distance, 2)} inches\n'
                # elif obj_centre < 3 * (frame_width / 5):
                #     text += f'The {d[0]} is at centre at {round(distance, 2)} inches\n'
                # elif obj_centre < 4 * (frame_width / 5):
                #     text += f'The {d[0]} is at slightly right at {round(distance, 2)} inches\n'
                # else:
                #     text += f'The {d[0]} is at right at {round(distance, 2)} inches\n'

                cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), (0, 0, 0), -1)
                cv.putText(frame, f'Dis: {round(distance, 2)}', (x + 5, y + 13), cv.FONT_HERSHEY_COMPLEX, 0.48,
                           (0, 255, 0), 2)

            _, jpeg = cv.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            global_text = text

        #     if time.time() - last_check_time >= 3:
        #         # socketio.emit('play_audio', {'text': text})
        #         play_audio(text)
        #         last_check_time = time.time()
        #     text = "Hello"
        #     cv.imshow("Frame", frame)
        #
        #     if cv.waitKey(1) == ord("q"):
        #         break
        # cap.release()
        # cv.destroyAllWindows()


# def generate_frames2():
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             data = object_detector(frame)
#             frame_width = frame.shape[1]
#             obj_centre = 0
#
#             def play_audio(text):
#                 engine.say(text)
#                 engine.runAndWait()
#
#             for d in data:
#                 text=""
#                 x, y = d[2]
#
#                 if d[0] == 'person':
#                     distance = distance_measure(focal_person, person_width, d[1])
#                     x, y = d[2]
#                     obj_centre = x + d[1] / 2
#                     if obj_centre < frame_width / 5:
#                         text += f'The {d[0]} is at left at {round(distance, 2)} inches\n'
#                     elif obj_centre < 2 * (frame_width / 5):
#                         text += f'The {d[0]} is at slightly left at {round(distance, 2)} inches\n'
#                     elif obj_centre < 3 * (frame_width / 5):
#                         text += f'The {d[0]} is at centre at{round(distance, 2)} inches\n'
#                     elif obj_centre < 4 * (frame_width / 5):
#                         text += f'The {d[0]} is at slightly right at {round(distance, 2)} inches\n'
#                     else:
#                         text += f'The {d[0]} is at right at{round(distance, 2)} inches\n'
#                 if d[0] == 'cell phone':
#                     distance = distance_measure(focal_mobile, mobile_width, d[1])
#                     x, y = d[2]
#                     obj_centre = x + d[1] / 2
#                     if obj_centre < frame_width / 5:
#                         text += f'The {d[0]} is at left at {round(distance, 2)} inches\n'
#                     elif obj_centre < 2 * (frame_width / 5):
#                         text += f'The {d[0]} is at slightly left at {round(distance, 2)} inches\n'
#                     elif obj_centre < 3 * (frame_width / 5):
#                         text += f'The {d[0]} is at centre at {round(distance, 2)} inches\n'
#                     elif obj_centre < 4 * (frame_width / 5):
#                         text += f'The {d[0]} is at slightly right at {round(distance, 2)} inches\n'
#                     else:
#                         text += f'The {d[0]} is at right at {round(distance, 2)} inches\n'
#
#                 cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), (0, 0, 0), -1)
#                 cv.putText(frame, f'Dis: Sahi Hai inch', (x + 5, y + 13), cv.FONT_HERSHEY_COMPLEX, 0.48, (0, 255, 0), 2)
#
#             _, jpeg = cv.imencode('.jpg', frame)
#             frame_bytes = jpeg.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
#
#
#             # if time.time() - last_check_time >= 3:
#             #     audio_thread = Thread(target=play_audio, args=(text,))
#             #     audio_thread.start()
#             #     last_check_time = time.time()
#             # text = ""

@app.route('/call_example_function')
def call_example_function():
    result = global_text
    return result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def handle_connect():
    print('Client connected')


if __name__ == '__main__':
    app.run(debug=True)
