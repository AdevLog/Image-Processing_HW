"""
Template matching using OpenCV
https://theailearner.com/tag/cv2-matchtemplate/
"""
import cv2
import numpy as np
 
# Read the input and template image
img = cv2.imread('D:/downloads/hearts_8.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread('D:/downloads/hearts_8_temp.png',0)
w, h = template.shape[::-1]
 
# Apply template matching
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
 
# Threshold the result
threshold = 0.95
loc = np.where(res >= threshold)
 
# Draw the rectangle
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (255,0,0), 1)
 
# Display the result
cv2.imshow('a',img)
cv2.waitKey(0)