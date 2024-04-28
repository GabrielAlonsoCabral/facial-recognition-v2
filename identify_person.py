import cv2
from deepface import DeepFace
from utils import GREEN, get_allowed_people, PersonInfo
from threading import Thread


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

counter = 0

allowed_people: list[PersonInfo] = get_allowed_people("./images")


def check_face(frame: cv2.typing.MatLike, person: PersonInfo):
    try:
        result = DeepFace.verify(
            frame, person.get("photo").copy(), model_name='Facenet')

        if result['verified']:
            person['verified'] = True
        else:
            person['verified'] = False
    except ValueError:
        person['verified'] = False


def threading_check_face(frame: cv2.typing.MatLike, person: PersonInfo):
    global counter

    if counter % 90 == 0:
        try:
            Thread(target=check_face,
                   args=(frame.copy(), person)).start()
        except ValueError:
            pass
    counter += 1


def detect_faces(frame: cv2.typing.MatLike):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:

        for person_info in allowed_people:
            print(f"Verifying {person_info.get("name")}")
            threading_check_face(
                frame.copy(), person_info)

            print(f"is_verified: {person_info.get("verified")}, name:{
                  person_info.get("name")}")

            if person_info.get("verified"):
                cv2.rectangle(frame, (x, y), (x+w, y+h), GREEN, 2)
                cv2.putText(frame, person_info.get("name"), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 1)


def must_stop():
    key = cv2.waitKey(1)

    if key == ord("q"):
        return True

    return False


while True:
    ret, frame = cap.read()

    if not ret:
        print("WEBCAM NOT CONNECTED")
        break

    detect_faces(frame)

    cv2.imshow("video", frame)

    if must_stop():
        break

cv2.destroyAllWindows()
