import cv2
from imageai.Detection import ObjectDetection
import os
import imutils
from imutils.video import FPS
cascPath = "haarcascade_frontalface_default.xml"
video_capture = cv2.VideoCapture(0)
execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
index = 0
timer = 0
while True:
    timer = timer + 1
    index = index + 1
    fps = FPS().start()
    ret,frame = video_capture.read()
    name = str(index) + '.jpg'
    cv2.imwrite(name, frame)
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, name), output_image_path=os.path.join(execution_path, name))
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor= 1.1,
        minNeighbors=5,
        minSize=(30, 30)
)
    if timer == 100000000000000000000000000000000000:
        name = str(index) + '.jpg'
        cv2.imwrite(name, frame)
        detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, name),output_image_path=os.path.join(execution_path, name))
        timer = 0
        index = index + 1
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        fps.update()
    #print("[INFO] elapsed time:[:.2f}".format(fps.elapsed()))
    #print("[INFO] approx.FPS:[:.2f}".format(fps.fps()))
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
