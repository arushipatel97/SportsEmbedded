import cv2
import numpy as np

original = cv2.imread('pro/pro1.jpg')
height, width, depth = original.shape
ratio = height/width
nHeight = 500
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
res = cv2.dilate(res2, None, iterations=1)
res = cv2.erode(res, None, iterations=1)
# res = cv2.Canny(res2,50,150,apertureSize = 5)

# res = res2

minLineLength = 30
maxLineGap = 10
lines = cv2.HoughLinesP(res2,1,np.pi/180,18,minLineLength,maxLineGap)
# lines = cv2.HoughLines(res,1,np.pi/180,80)
for line in lines:
    for x1,y1,x2,y2 in line:
    	cv2.line(disp,(x1,y1),(x2,y2),(0,255,0),1)

cv2.imshow('res',res)
cv2.imshow('res1',res1)
cv2.imshow('res2',res2)
cv2.imshow('original',img)
# cv2.imshow('threshold',th3)
cv2.imshow('hsv',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# g = cv2.cvtColor(green_image, cv2.COLOR_BGR2GRAY)
# (channel_b, channel_g, channel_r) = cv2.split(img)
# grayscaled = cv2.GaussianBlur(grayscaled,(1,1),0)
# th = cv2.adaptiveThreshold(channel_g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 1)

# threshold = 6
# minLineLength = 10
# lines = cv2.HoughLinesP(th3, 1, np.pi/180, threshold, 0, minLineLength, 20);