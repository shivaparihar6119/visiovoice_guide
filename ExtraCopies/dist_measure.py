import cv2 as cv
import time
import numpy as np
import pyttsx3
from threading import Thread

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

known_dist=45
known_width=14.3
person_width=16.0#inches
mobile_width=3 #inches
book_width=16
ball_width=2.8
bottle_width=2.5

CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.5
fonts = cv.FONT_HERSHEY_COMPLEX
capt = cv.VideoCapture(0)
# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

class_names = []

with open("classes.txt", "r") as f:
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
        elif classid == 73:
            data_list.append([class_names[int(classid)], box[2], (box[0], box[1] - 2)])
        elif classid == 32:
            data_list.append([class_names[int(classid)], box[2], (box[0], box[1] - 2)])
        elif classid == 39:
            data_list.append([class_names[int(classid)], box[2], (box[0], box[1] - 2)])


    return data_list

def focal_length(measured_distance, real_width, width_in_rf_image):

    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value

def distance_measure(focal_length, real_face_width, face_width_in_frame):
    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance

ref_person = cv.imread('image14.png')
ref_mobile = cv.imread('image4.png')
ref_book =cv.imread('image5.png')
ref_ball=cv.imread('image2.png')
ref_bottle=cv.imread('image0.png')

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

book_data = object_detector(ref_book)
book_width_in_rf = book_data[0][1]

ball_data = object_detector(ref_ball)
ball_width_in_rf = ball_data[0][1]

bottle_data = object_detector(ref_bottle)
bottle_width_in_rf = bottle_data[0][1]

print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf} book width in pixel: {book_width_in_rf}")
focal_person = focal_length(known_dist, person_width, person_width_in_rf)

focal_mobile = focal_length(known_dist, mobile_width, mobile_width_in_rf)

focal_book = focal_length(known_dist, book_width, book_width_in_rf)

focal_ball = focal_length(known_dist, ball_width, ball_width_in_rf)

focal_bottle = focal_length(known_dist, bottle_width, bottle_width_in_rf)

last_check_time = time.time()


def play_audio(text):
    engine.say(text)
    engine.runAndWait()


while True:
   ret,frame=capt.read()
   data=object_detector(frame)
   frame_width=frame.shape[1]
   frame_height = frame.shape[0]

   obj_centre=0
   for d in data:
       text=""
       if d[0] == 'person':
           distance = distance_measure(focal_person, person_width, d[1])
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
               position_text += 'centre'
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

       if d[0] == 'cell phone':
           distance = distance_measure(focal_mobile, mobile_width, d[1])
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
               position_text += 'centre'
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

       if d[0] == 'book':
           distance = distance_measure(focal_book, book_width, d[1])
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
               position_text += 'centre'
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

       if d[0] == 'sports ball':
           distance = distance_measure(focal_ball, ball_width, d[1])
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
               position_text += 'centre'
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

       if d[0] == 'bottle':
           distance = distance_measure(focal_ball, ball_width, d[1])
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
               position_text += 'centre'
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


       cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
       cv.putText(frame, f'Dis: {round(distance, 2)} inch', (x + 5, y + 13), fonts, 0.48, GREEN, 2)
   if time.time() - last_check_time >= 3:
       audio_thread = Thread(target=play_audio, args=(text,))
       audio_thread.start()
       last_check_time = time.time()
       text=""

   cv.imshow("Frame",frame)

   if cv.waitKey(1)==ord("q"):
       break

capt.release()
cv.destroyAllWindows()






