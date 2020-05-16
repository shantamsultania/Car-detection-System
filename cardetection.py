import cv2

# for a video

# videoCapture("name of video .mp4")
cam = cv2.VideoCapture("video1.mp4")
cas = cv2.CascadeClassifier('cars.xml')

while True:
    check,frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_de = cas.detectMultiScale(gray, 1.1, 3)

    for x, y, w, h in face_de:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("face detected", frame)
    # cv2.imshow("real-time detection",frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()