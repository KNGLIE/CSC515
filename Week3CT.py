import cv2
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from matplotlib import pyplot as plt
                  
img = Image.open(BytesIO(requests.get('https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSulLs5l2Bwr6iFywEqHxQWevj9snjLRrsPxjQWsCIrJmA9cT4q').content))                                 
image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) 
gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)


img_2 = Image.open(BytesIO(requests.get('https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQdkf_IPn6VW3jpJ_fTU4IUcwtGPbcvSfxXW4EPXjeVuMjM1Baz').content))                                 
image_2 = cv2.cvtColor(np.array(img_2), cv2.COLOR_RGB2BGR)


rows, cols = image.shape[:2]
# setting up to crop images
T = np.float32([[1,0,-150], [0,1,0]])
R = np.float32([[1,0, 150], [0,1,0]]) 
# setting up rotations
rotate = cv2.getRotationMatrix2D((cols/2,rows/2), -10, 1)
rotate2 = cv2.getRotationMatrix2D((cols/2,rows/2), -3, 1)
# cropping/rotating images
trans_img = cv2.warpAffine(image, T, (cols, rows))
trans_img2= cv2.warpAffine(image, R, (cols, rows))
rotate_img = cv2.warpAffine(trans_img, rotate2,(cols, rows))
trans_image = cv2.warpAffine(image_2, T, (cols, rows))
trans_image2= cv2.warpAffine(image_2, R, (cols, rows))
rotate_image = cv2.warpAffine(trans_image2, rotate,(cols, rows))

# splitting individual channels to find ideal lighting
b,g,r = cv2.split(rotate_img)
#bgr_split = np.concatenate((b,g,r), axis=1)
#cv2.imshow('bgr', r)
a,b,c = cv2.split(rotate_image)
#bgr_split = np.concatenate((a,b,c), axis=1)
#cv2.imshow('1',a)
d,e,f = cv2.split(trans_image)
#bgr_split = np.concatenate((d,e,f), axis=1)
#cv2.imshow('', f)
g,h,i = cv2.split(trans_img2)
#bgr_split = np.concatenate((g,h,i), axis=1)
#cv2.imshow('', h)
# resizing part of the image
stretch_near = cv2.resize(f, (300, 173), interpolation = cv2.INTER_LINEAR)

images = [a, stretch_near, r, h]
# displaying image in plot without black removed
for i in range(4):
	plt.subplot(2,2,i+1)
	plt.imshow(images[i])
plt.show()

# concatenating images into one
conc = np.concatenate((a, stretch_near, r, h), axis=1)
h, w = conc.shape[:2]
first_pass = True
pixels = np.sum(conc, axis=0).tolist()
# building image with limited black space
for index, value in enumerate(pixels):
	if value == 0:
		continue
	else:
		ROI = conc[0:h, index:index+1]
		if first_pass:
			result = conc[0:h, index+1:index+2]
			first_pass = False
			continue
		result = np.concatenate((result, ROI), axis=1)


cv2.imshow('conc', result)

cv2.imwrite('week3.jpg', result)
cv2.waitKey(0) 
cv2.destroyAllWindows()