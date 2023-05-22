import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

image = cv2.imread('Mod4CT1.jpg')
#cv2.imwrite('Mod4CT1_copy.png', image)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


w,h = image.shape[:2]
print(w,h)
sigma = math.sqrt(w*h)
sigma2 = w*h
median = cv2.medianBlur(image, 3)
mean = cv2.blur(image,(3, 3))
gaussian = cv2.GaussianBlur(image,(3, 3), sigma)
gaussian_2 = cv2.GaussianBlur(image,(3, 3), sigma2)

median_5 = cv2.medianBlur(image, 5)
mean_5 = cv2.blur(image,(5, 5))
gaussian_5 = cv2.GaussianBlur(image,(5 ,5), sigma)
gaussian_5_2 = cv2.GaussianBlur(image,(5 ,5), sigma2)

median_7 = cv2.medianBlur(image, 7)
mean_7 = cv2.blur(image,(7, 7))
gaussian_7 = cv2.GaussianBlur(image,(7,7), w%h*h)
gaussian_7_2 = cv2.GaussianBlur(image,(9,9), sigma2)


titles = ['mean','median', 'gaussian1', 'gaussian2']
images = [mean, median, gaussian, gaussian_2, mean_5, median_5, gaussian_5, gaussian_5_2, mean_7, median_7, gaussian_7, gaussian_7_2]
count = 0

for i in range(len(images)):
	
	if count <= 4:
		count += 1
		plt.title(titles[i-1])
	plt.subplot(3, 4, i + 1), 
	plt.imshow(images[i])
	
	
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
