import threading
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

counter = 0

face_match = False

reference_img = cv2.imread("images/gabriel/74.jpg")


def check_face(frame):
    global face_match
    try:

        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False


while True:

    ret, frame = cap.read()

    if not ret:
        print("WEBCAM NOT CONNECTED")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if counter % 30 == 0:
        try:
            threading. Thread(target=check_face,
                              args=(frame.copy(),)).start()
        except ValueError:
            pass

    counter += 1

    if face_match:
        cv2.putText(frame, "MATCH!", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    else:
        cv2.putText(frame, "NO MATCH!", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow("video", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break
cv2.destroyAllWindows()
