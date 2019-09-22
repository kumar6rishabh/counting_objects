import cv2
import numpy as np

img = cv2.imread('water_coins.jpg')
gr_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
_ , thresh = cv2.threshold(gr_img , 0 , 255 , cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#noise removal

kernel = np.ones((3 , 3) , np.uint8)
opening = cv2.morphologyEx(thresh , cv2.MORPH_OPEN , kernel , iterations = 2)

#Sure background

kernel = np.ones((3 , 3) , np.uint8)
sure_bg = cv2.dilate(opening , kernel , iterations = 3)

#FINDING SURE FOREGROUND AREA

dist_transform = cv2.distanceTransform(opening , cv2.DIST_L2 , 5)
ret , sure_fg = cv2.threshold(dist_transform , 0.7*dist_transform.max() , 255 , 0)

#finding unknown region

#cv2.imshow('sure_fg0' , sure_fg)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg , sure_fg)
#cv2.imshow('sure_fg' , sure_fg)
#cv2.imshow('sure_bg' , sure_bg)

ret , markers = cv2.connectedComponents(sure_fg)
for i in range(markers.shape[0]):
    print(set(markers[i]))
#cv2.imshow('here' , markers)
markers = markers + 1

markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

#cv2.imshow('img' , img)
