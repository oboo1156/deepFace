import cv2 as cv

classifier = cv.CascadeClassifier('haarcascade_frontalface_default2.xml')


def detect_face(img, draw_box=True):
    """ a function to detect face, returns image(face and background)
    together with box as img and boxed face alone as face_box """

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))

    face_box = None
    for (x, y, w, h) in faces:
        if draw_box:
            cv.rectangle(img, (x, y),  (x+w, y+h), (255, 0, 0), 3)

        face_box = img[y:y+h, x:x+w]

    return img, face_box
