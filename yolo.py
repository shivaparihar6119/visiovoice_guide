import cv2 as cv

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.3
fonts = cv.FONT_HERSHEY_COMPLEX

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
