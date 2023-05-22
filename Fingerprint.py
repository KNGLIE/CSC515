import cv2
import numpy as np

path = r"C:\Users\lance\Desktop\CSC 515 Python\fingerprint.png"

img = cv2.imread(path, 0)

kernel = np.ones((2,2), np.uint8)

erosion = cv2.erode(img, kernel, iterations=1)

dilation = cv2.dilate(img, kernel, iterations=1)

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow('open', opening)

cv2.imshow('close', closing)

cv2.imshow('erode', erosion)

cv2.imshow('dilate', dilation)

cv2.imwrite('Open print.png', opening)
cv2.imwrite('Close print.png', closing)
cv2.imwrite('Erode print.png', erosion)
cv2.imwrite('Dilate print.png', dilation)

cv2.waitKey(0)
cv2.destroyAllWindows()