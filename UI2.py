import cv2
import numpy as np
import PySimpleGUI as sg
import time
from playsound import playsound as play
sg.theme("LightGreen")
layout = [
        [sg.Text("STUD-bot v1", size=(60, 1), justification="center")],
        [sg.Image(filename="", key="-IMAGE-")],
        [
            sg.Text('Max break Time in Minutes', size=(10, 1)),
            sg.Slider(
                (1, 30),
                0,
                1,
                orientation="h",
                size=(40, 15),
                key="-MAX BREAK TIME-",
            ),
        ],
        [
            sg.Text("Eye Aspect Ratio",  size=(10, 1)),
            sg.Slider(
                (0, 100),
                0,
                1,
                orientation="h",
                size=(40, 15),
                key="-EYE_AR_THRESH-",
            )
        ],
        [
            sg.Text("Consequitive frames",  size=(10, 1), ),
            sg.Slider(
                (0, 60),
                0,
                1,
                orientation="h",
                size=(40, 15),
                key="-EYE_AR_CONSEC_FRAMES-",
            )
        ],
        [sg.Button("Exit", size=(10, 1))],
    ]
window = sg.Window("Stud-bot", layout, location=(800, 400))
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")

classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]




layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


colors= np.random.uniform(0,255,size=(len(classes),3))




from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2



def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear


args = {"shape_predictor": "shape_predictor_68_face_landmarks.dat", "alarm": "alarm.wav", "webcam": 0}


EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 15


COUNTER = 0
ALARM_ON = False

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")

vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

frame_id=0
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
break_time=0

while True:
    person_detected=False
    frame_start_time=time.time()
    event, values = window.read(timeout=20)
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    EYE_AR_THRESH = values["-EYE_AR_THRESH-"]/100
    EYE_AR_CONSEC_FRAMES = values["-EYE_AR_CONSEC_FRAMES-"]

    frame_id += 1

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width, channels = frame.shape

    rects = detector(gray, 0)

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)  # reduce 416 to 320
    net.setInput(blob)
    outs = net.forward(outputlayers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(
                    float(confidence))
                class_ids.append(class_id)
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)


    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if(label=='person'):
                person_detected=True

            if (label == "cell phone"):
                play('alarm.wav')
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, (255, 255, 255), 2)

    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:

                if not ALARM_ON:
                    ALARM_ON = True


                    play(args["alarm"])

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            ALARM_ON = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS:" + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 1)

    #cv2.imshow("Frame", frame)
    #key = cv2.waitKey(1) & 0xFF

    if (person_detected==False):
        break_time+=time.time()-frame_start_time
        cv2.putText(frame, "Break Time" + str(break_time//60)+":"+str(round(break_time%60,2)), (10, 100), font, 2, (0, 0, 0), 1)
        if break_time>values["-MAX BREAK TIME-"]*60:
            play('alarm.wav')
    else:
        break_time=0

    imgbytes = cv2.imencode(".png", frame)[1].tobytes()
    window["-IMAGE-"].update(data=imgbytes)

    #if key == ord("q"):
    #    break



cv2.destroyAllWindows()
window.close()
vs.stop()