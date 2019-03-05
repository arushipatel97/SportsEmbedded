import cv2
import numpy as np
import math

def slope(x1, y1, x2, y2):
    m = 0
    b = (x2 - x1)
    d = (y2 - y1)
    if b != 0:
        m = (d)/(b) 

    return m

original = cv2.imread('pro/pro5.jpg')
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
res5 = cv2.adaptiveThreshold(res1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,3,-5)
res5 = cv2.blur(res5,(3,3),0)
r = cv2.dilate(res5, None, iterations=1)
r = cv2.blur(r,(5,5))
r = cv2.Canny(r, 100, 170, apertureSize = 3)
r = cv2.erode(r, None, iterations=0)
r = cv2.dilate(r, None, iterations=2)
r = cv2.erode(r, None, iterations=3)
r = cv2.Canny(r, 100, 170, apertureSize = 3)

res = cv2.dilate(res5, None, iterations=4)
res = cv2.erode(res, None, iterations=2)

res2 = cv2.dilate(res5, None, iterations=1) 

lines = cv2.HoughLinesP(res2,rho = 1,theta = 1*np.pi/180,threshold = 100,minLineLength = 100,maxLineGap = 10)
count = 0
slopes = {0}
print(len(lines))
for line in lines:
    for x1,y1,x2,y2 in line:
    	diff = True
    	m = slope(x1, y1, x2, y2)
    	# print(m)
    	dist = math.hypot(x2 - x1, y2 - y1)
    	if dist > 10 and abs(m) > 0.00:
    		for s in slopes:
    			if (abs(s-m) < 0.08):
    				print(m)
    				diff = False
    		if (diff == True):
	    		count+=1
	    		slopes.add(m)
	    		cv2.line(disp,(x1,y1),(x2,y2),(255,0,0),2)

print(slopes)
print(len(slopes))
cv2.imshow('res',res)
cv2.imshow('res1',res1)
cv2.imshow('res2',res2)
cv2.imshow('original',img)
# cv2.imshow('threshold',th3)
cv2.imshow('hsv',mask)
cv2.waitKey(0)
cv2.imwrite("r.png", r)
cv2.imwrite("res1.png", res1)
cv2.imwrite("final.png", disp)
cv2.imwrite("res2.png", res2)
cv2.imwrite("res5.png", res5)
cv2.imwrite("res.png", res)
cv2.destroyAllWindows()