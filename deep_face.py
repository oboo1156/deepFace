import cv2 as cv
import face_detection
from deepface import DeepFace

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame, face_box = face_detection.detect_face(frame)
    if face_box is not None:
        prediction = DeepFace.analyze(frame, actions=('emotion',))
        cv.putText(face_box, text=prediction['dominant_emotion'], org=(00, 185), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv.LINE_AA, bottomLeftOrigin=False)

    cv.imshow('Video', frame)  # loop the images in the form of video, together with the detected face(frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
cv.waitKey(1)

