import cv2

image = cv2.imread('Class.jpg')
cv2.imwrite('Class_copy.png', image)
cv2.imshow('Class_window', image)
cv2.waitKey(0)
cv2.destroyAllWindows()