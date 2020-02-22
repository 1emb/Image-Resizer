# USAGE
# python click_and_crop.py --image jurassic_park_kitchen.jpg

# import the necessary packages
import argparse
import cv2
import numpy as np
import math
import os

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
num = 0

def root(a):
    return math.sqrt(a)

def length(a, b):
    return math.sqrt((b[1]-a[1])**2 + (b[0]-a[0])**2)

def corner(c, a, b):
    if (length(c,a)*length(c,b) == 0):
        return 0
    elif ((length(b,c)**2+length(c,a)**2-length(a,b)**2)/(2*length(c,a)*length(c,b))) >=1:
#        return round(np.arccos(2-((length(b,c)**2+length(c,a)**2-length(a,b)**2)/(2*length(c,a)*length(c,b))))/np.pi*180, 2)
        print((length(b,c)**2+length(c,a)**2-length(a,b)**2)/(2*length(c,a)*length(c,b)), c,a,b)
        return 180
    else:
        return abs(round(np.arccos((length(b,c)**2+length(c,a)**2-length(a,b)**2)/(2*length(c,a)*length(c,b)))/np.pi*180, 2))
                
    
def getFunc(x, y):
    a = round((x[1]-y[1])/(x[0]-y[0]))
    b = x[1]-x[0]*a
    return [a, b]

def vectorCheck(a, b, x, y):
    if y == a*x+b:
        return True
    else:
        return False
    
def inside(x, a, b):
    if x>a and x>b:
        return False
    if x<a and x<b:
        return False
    return True
    

def imageProcess():
    try:
        os.remove("resukt.jpg")
    except:
        pass
    A = refPt[0]
    B = refPt[1]
    C = refPt[2]
    D = refPt[3]
    size = [A[0], A[1], A[0], A[1]]
    W = [A, C]
    H = [B, D]
    for i in range(1, 4):
        if (refPt[i][0]<size[0]):
            size[0] = refPt[i][0]
        if (refPt[i][0]>size[1]):
            size[1] = refPt[i][0]
        if (refPt[i][1]<size[2]):
            size[2] = refPt[i][1]
        if (refPt[i][1]>size[3]):
            size[3] = refPt[i][1]
        
    print(size)
    print(refPt)
        
    blank_image = np.zeros((abs(size[3]-size[2]), abs(size[1]-size[0]), 3),np.uint8)
    Hdiff = (size[3]+size[2]-abs(size[3]-size[2]))//2
    Wdiff = (size[1]+size[0]-abs(size[1]-size[0]))//2
    print(abs(size[3]-size[2]), abs(size[1]-size[0]))
    
    for i in range(0,abs(size[3]-size[2])):
        for j in range(0, abs(size[1]-size[0])):
            blank_image[i,j] = (255,255,255)
#    for i in range(0,abs(size[3]-size[2])):
#        for j in range(0, abs(size[1]-size[0])):
#            E = [i+Hdiff,j+Wdiff]
#            
#            if (corner(E, A, B)+ corner(E, B, C) + corner(E, C, D)+corner(E, D, A)) != 360:
#                blank_image[i,j] = image[i+Hdiff,j+Wdiff]
        
    cv2.imwrite('resukt.jpg', blank_image)

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
    global refPt, cropping, num
        
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(refPt) <4:
            refPt.append((x, y))
            cropping = True
            print(refPt[num])
            
            print(image.shape)
            num += 1
        else:
            imageProcess()
            num = 0
            refPt = []
            
	# check to see if the left mouse button was released
#	elif event == cv2.EVENT_LBUTTONUP:
#		# record the ending (x, y) coordinates and indicate that
#		# the cropping operation is finished
#		refPt.append((x, y))
#		cropping = False
#
#		# draw a rectangle around the region of interest
#		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
#		cv2.imshow("image", image)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()

	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break

# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	cv2.imshow("ROI", roi)
	cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()