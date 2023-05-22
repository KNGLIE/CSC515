import cv2


def face_border(picture):
    img = cv2.imread(picture)
    img_x, img_y = img.shape[:2]
    while img_y + img_x >= 2000:
        img = cv2.resize(img, (0, 0), fx=.5, fy=.5)
        img_x, img_y = img.shape[:2]
    while img_y + img_x <= 2000:
        img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5)
        img_x, img_y = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.shape)

    path = 'haarcascade_frontalface_default.xml'
    path_to_eyes = "haarcascade_eye.xml"
    face_cascade = cv2.CascadeClassifier(path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.45, minNeighbors=5, minSize=(15, 15), flags=(cv2.CASCADE_SCALE_IMAGE + cv2.CASCADE_DO_CANNY_PRUNING))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eye_cascade = cv2.CascadeClassifier(path_to_eyes)

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.15, minNeighbors=3, minSize=(5, 5))

        for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 255), 1)
            roi = roi_color.copy()
            roi = roi[ey:ey + eh, ex:ex + ew]
            blur = cv2.medianBlur(roi, 11)
            blur = cv2.medianBlur(blur, 11)
            blur = cv2.medianBlur(blur, 11)
            blur = cv2.medianBlur(blur, 11)
            blur = cv2.medianBlur(blur, 11)

            roi_color[ey:ey + eh, ex:ex + ew] = blur

    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('copy_'+picture, img)



face_border('val.jpg')
face_border('fam.jpg')
face_border('chan.jpg')
face_border('chanplu.jpg')
