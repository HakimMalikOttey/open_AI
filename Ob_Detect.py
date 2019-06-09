import cv2
#from picamera.array import PiGBArray
from imageai.Detection import VideoObjectDetection
import os
import argparse
import imutils
from imutils.video import FPS

execution_path = os.getcwd()
detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()
video_capture = cv2.VideoCapture(0)
#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default= 300, help="minimum area size")
args = vars(ap.parse_args())
# initialize the first frame in the video stream. Do no place in While True, as that would deactivate object recognition
firstframe = None
while True:
    fps = FPS().start()
    ret,frame = video_capture.read()
    #Window size and image blur
    frame = imutils.resize(frame, width = 400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    #If the first frame is none, initialize it
    if firstframe is None:
        firstframe = gray
        continue
    frameDelta = cv2.absdiff(firstframe,gray)
    thresh = cv2.threshold(frameDelta,25,255,cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh,None,iterations=2)
    contour = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)

    for c in contour:
        #If countour is too small
        if cv2.contourArea(c) < args["min_area"]:
            continue
        # draws box around detected object
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame,(x,y), ( x + w, y +h), (0,225,0),2)
        #detections = detector.detectObjectsFromVideo(camera_input = video_capture,output_file_path=os.path.join(execution_path, "camera_detected_1"),frames_per_second=29, log_progress=True)
    cv2.imshow("Video", frame)
