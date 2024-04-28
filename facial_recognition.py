import threading
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

counter = 0

face_match = False

reference_img = cv2.imread("images/gabriel/80.jpg")

GREEN = (0, 255, 0)
RED = (0, 0, 255)


def check_face(frame: cv2.typing.MatLike):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy(), model_name='Facenet')['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False


def detect_faces(frame: cv2.typing.MatLike):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        if face_match:
            cv2.rectangle(frame, (x, y), (x+w, y+h), GREEN, 2)
            cv2.putText(frame, "Matched", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 1)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), RED, 2)
            cv2.putText(frame, "Not matched", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 1)


def must_stop():
    key = cv2.waitKey(1)

    if key == ord("q"):
        return True

    return False


def on_face_match():
    cv2.putText(frame, "MATCH!", (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 2, GREEN, 3)


def on_face_unmatch():
    cv2.putText(frame, "NO MATCH!", (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 2, RED, 3)


while True:
    ret, frame = cap.read()

    detect_faces(frame)

    if not ret:
        print("WEBCAM NOT CONNECTED")
        break

    if counter % 30 == 0:
        try:
            threading. Thread(target=check_face,
                              args=(frame.copy(),)).start()
        except ValueError:
            pass

    counter += 1

#    if face_match:
#        on_face_match()

#    else:
#        on_face_unmatch()

    cv2.imshow("video", frame)

    if must_stop():
        break

cv2.destroyAllWindows()
