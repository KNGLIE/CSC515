import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D


def main(x):
    img = cv2.imread(x)
    img = cv2.resize(img, (960, 540))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1, vmax=1)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    #fig = plt.figure()
    #axis = fig.add_subplot(1, 1, 1, projection="3d")
    #axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker='.')
    #axis.set_xlabel('Hue')
    #axis.set_ylabel('Sat')
    #axis.set_zlabel('val')
    #plt.show()


    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    blur_2 = cv2.GaussianBlur(blur, (7, 7), 0)
    blur_3 = cv2.GaussianBlur(blur_2, (3, 3), 0)
    blur_3 = cv2.GaussianBlur(blur_3, (3, 3), 0)
    blur_3 = cv2.GaussianBlur(blur_3, (3, 3), 0)

    adapt = cv2.adaptiveThreshold(blur_3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    adapt_2 = cv2.adaptiveThreshold(blur_3, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    cv2.imshow('adapt2', adapt_2)
    cv2.imshow(x, adapt)
    cv2.imwrite('Gaussian'+x, adapt)
    cv2.imwrite('Mean'+x, adapt_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



main('Pic1.jpg')
main('Pic2.jpg')
