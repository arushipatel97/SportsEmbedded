import cv2
import numpy as np
import math

original = cv2.imread('pro/pro1.jpg')
height, width, depth = original.shape
ratio = height/width
nHeight = 1000
nWidth = nHeight/ratio
img = cv2.resize(original,(int(nWidth),int(nHeight)))

line_image = np.copy(img)*0 #creating a blank to draw lines on
disp = np.copy(img)*0 #creating a blank to draw lines on
grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
grayscaled = cv2.dilate(grayscaled, None, iterations=1)

# look for green portion
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
greenLower = (29, 46, 26)
greenUpper = (64, 255, 255)
mask = cv2.inRange(hsv, greenLower, greenUpper)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)

im2,contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
if len(contours) != 0:
	# isolate field
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    crop_img = img[y:y+h, x:x+w]
    disp = img[y:y+h, x:x+w]
    mask = mask[y:y+h, x:x+w]
    cv2.imshow("cropped", crop_img)
    gray_crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)

res1 = cv2.bitwise_and(gray_crop_img,gray_crop_img,mask = mask)
res2 = cv2.adaptiveThreshold(res1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,3,-5)
res = cv2.dilate(res2, None, iterations=3)
res = cv2.erode(res, None, iterations=1)
masked_edges = cv2.Canny(res1, 100, 170, apertureSize = 3)
masked_edges = cv2.dilate(masked_edges, None, iterations=3)
masked_edges = cv2.erode(masked_edges, None, iterations=4)

minLineLength = 100
maxLineGap = 20
lines = cv2.HoughLinesP(masked_edges,1,np.pi/180,180,minLineLength,maxLineGap)
# lines = cv2.HoughLines(res,1,np.pi/180,80)
print(len(lines))
for line in lines:
    for x1,y1,x2,y2 in line:
    	# dist = math.hypot(x2 - x1, y2 - y1)
    	# if dist > 10 :
    # x1,y1,x2,y2 = line[0]
    	cv2.line(disp,(x1,y1),(x2,y2),(255,0,0),2)

cv2.imshow('res',res)
cv2.imshow('res1',res1)
cv2.imshow('res2',res2)
cv2.imshow('original',img)
cv2.imwrite("final.png", disp)
cv2.imwrite("masked_edges.png", masked_edges)
cv2.imwrite("res1.png", res1)
cv2.imshow('masked_edges',masked_edges)
cv2.waitKey(0)