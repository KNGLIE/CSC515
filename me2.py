import numpy as np
import cv2

image = cv2.imread('mealso.jpg', 1)
img = cv2.resize(image, (0,0), fx=.2, fy=.2)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
path = "haarcascade_frontalface_default.xml"
path_to_eyes = "haarcascade_eye.xml"

face_cascade = cv2.CascadeClassifier(path)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(40,40))
print(len(faces))

for (x, y, w, h) in faces:
	cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
	cv2.putText(img, 'This is me', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

	eye_cascade = cv2.CascadeClassifier(path_to_eyes)
	gray_roi = gray[y:y+h, x:x+w]
	eyes = eye_cascade.detectMultiScale(gray_roi)
	
	color = img[y:y+h, x:x+w]
	for (ex, ey, ew, eh) in eyes:
		radius = eh//2
		eye_x = int(ex+.5*ew)
		eye_y = int(ey+.5*ey)
		cv2.circle(color, (eye_x,eye_y), radius, (0,255,0), 2)

cv2.imwrite('ThisIsMe.jpg', img)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
