# USAGE
# python click_and_crop.py --image jurassic_park_kitchen.jpg

# import the necessary packages
import argparse
import cv2
import numpy as np
import math
import os
import sys

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
cropping = False
# for controlling image input sequence
refPt = []
num = 0

def draw(A):
    cv2.circle(image, (A[0], A[1]), 7, (0,255,0), -1)
    cv2.imshow('image',image)

def line(A, B):
    cv2.line(image,(round(A[0]),round(A[1])),(round(B[0]),round(B[1])),(255,255,255),5)
    
def vectorLength(a):
    return math.sqrt(a[0]**2 + a[1]**2)

def alphaFinder(a, b, calculateBeta):        
    def checkAlphaBetaValid(alpha, beta):
        return (alpha >= 1) and (beta >= 0) and (beta <= 1)
    if a**2/4-b>=0:
        x1 = math.sqrt(a**2/4-b) -a/2
        x2 = -math.sqrt(a**2/4-b) -a/2
        if (x1 >= 1) and (x2 < 1):
            alpha = x1
            beta = calculateBeta(alpha)
            if checkAlphaBetaValid(alpha, beta):
                return [alpha, beta]
        elif (x1 < 1) and (x2 >= 1):
            alpha = x2
            beta = calculateBeta(alpha)
            if checkAlphaBetaValid(alpha, beta):
                return [alpha, beta]
        elif (x1 >= 1) and (x2 >= 1):
            alpha = x1
            beta = calculateBeta(alpha)
            if checkAlphaBetaValid(alpha, beta):
                return [alpha, beta]
            else:
                alpha = x2
                beta = calculateBeta(alpha)
                if checkAlphaBetaValid(alpha, beta):
                    return [alpha, beta]
        
def transformFunction(A, B, C, D):
    AB = [B[0]-A[0], B[1]-A[1]]
    AD = [D[0]-A[0], D[1]-A[1]]
    AMx = AB[0]/vectorLength(AB)
    AMy = AB[1]/vectorLength(AB)
    ANx = AD[0]/vectorLength(AD)
    ANy = AD[1]/vectorLength(AD)
    MI = np.linalg.inv(np.matrix([[AMx, ANx], [AMy, ANy]]))
    def transposeAndMI(point):
        point = [point[0] - A[0], point[1] - A[1]]
        return np.dot(MI, np.matrix(point).transpose()).transpose().tolist()[0]
#    Alpha = AlphaFinder(AGy*BCx+AGx*(ADy-BCy)+ABx*ADy, ABy+ABx*(ADy-BCy))
#    Beta = (Al*AGx-ABx)/BCx
    AAfter = transposeAndMI(A)
    BAfter = transposeAndMI(B)
    CAfter = transposeAndMI(C)
    DAfter = transposeAndMI(D)
    BCx = float(CAfter[0]-BAfter[0])
    ADy = float(DAfter[1]-AAfter[1])
    BCy = float(CAfter[1]-BAfter[1])
    ABx = float(BAfter[0]-AAfter[0])
    def calculateBeta(alpha):
        return (alpha*AGx-ABx)/BCx
    def transform(point):
        try:
            AG = transposeAndMI(point)
            AGx = AG[0]
            AGy = AG[1]
            a = (AGx*BCy-AGx*ADy-ABx*ADy-AGy*BCx)/(AGx*ADy)
            b = (ABx*ADy-ABx*BCy)/(AGx*ADy)
            result = alphaFinder(a, b, calculateBeta)
            if result == None:
                raise RuntimeError
            alpha = result[0]
            beta = result[1]
            return np.add(np.matrix([0, ADy]).transpose()*beta, np.matrix([ABx, 0]).transpose()/alpha).transpose().tolist()[0]
        except:
            return [0, 0]
    return transform

def Convolution(blank_image, kernel):
    def limit(number):
        return min(max(number, 0), 254)
    level = len(kernel)
    if level == 5:
        diff = 2
    elif level == 3:
        diff = 1
    else:
        raise RuntimeError
    print("start convo")
    width = blank_image.shape[1]
    height = blank_image.shape[0]
    new = np.zeros((height, width, 3), np.uint8)
    for x in range (0, width):
        for i in range(0, diff + 1):            
            new[i, x] = blank_image[i, x]
            new[height-i-1, x] = blank_image[height-i-1, x]        
    for y in range (0, height):
        for i in range(0, diff + 1):
            new[y, i] = blank_image[y, i]
            new[y, width-i-1] = blank_image[y, width-i-1]
    for y in range (diff, height-diff):
        for x in range (diff, width-diff):
            sums = [0, 0, 0]
#            sumR = 0
#            sumG = 0
#            sumB = 0
            for i in range(-diff, diff + 1):
                for j in range(-diff, diff + 1):
                    sums = [sums[w] + blank_image[y+i, x+j][w] * kernel[j+1][i+1] for w in range(0, 3)]
#                    sumR += blank_image[y+i, x+j][2]*kernel[j+1][i+1]
#                    sumG += blank_image[y+i, x+j][1]*kernel[j+1][i+1]
#                    sumB += blank_image[y+i, x+j][0]*kernel[j+1][i+1]
#            new[y, x] = [limit(sumB), limit(sumG), limit(sumR)]
            new[y, x] = [limit(aSum) for aSum in range(0, 3)]
    return new

def normalizeKernel(kernel, factor):
    for i in range(len(kernel)):
        for j in range(len(kernel[0])):
            kernel[i][j] = kernel[i][j] * factor
    return kernel

def imageProcess():
    try:
        os.remove("resukt.jpg")
    except:
        pass
    A = refPt[0]
    B = refPt[1]
    C = refPt[2]
    D = refPt[3]
    
    transformF = transformFunction(A, B, C, D)
    resizeRatio=0.9
    BD = [1.0*D[0]-B[0], 1.0*D[1]-B[1]]
    BDLength = vectorLength(BD)
    BD=[BD[0]/BDLength, BD[1]/BDLength]
    O = [BD[0]+B[0], BD[1]+B[1]]
    I = [-BD[0]+D[0], -BD[1]+D[1]]
    draw([round(O[0]), round(O[1])])
    draw([round(I[0]), round(I[1])])
    line(O, I)
    line(A, C)
    transformedO = round(2 + resizeRatio * transformF(O)[0])
    transformedI = round(2 + resizeRatio * transformF(I)[1])
    print("transformedO: ", transformF(O))
    blank_image = np.zeros((transformedI, transformedO, 3),np.uint8)
    for y in range(0, image.shape[0]):
        for x in range(0, image.shape[1]):
            newPixelLocation = transformF([x * 1.0, y * 1.0])
            blank_image[round(resizeRatio * newPixelLocation[1]), round(resizeRatio * newPixelLocation[0])] = image[y, x]
    gaussianBlur3 = [[1.0/16, 1.0/8, 1.0/16], [1.0/8, 1.0/4, 1.0/8], [1.0/16, 1.0/8, 1.0/16]]
    sharpen3 = [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]]
    emboss3 = [[-2, -1, 0],[-1, 1, 1],[0, 1, 2]]
    gaussianBlur5 = normalizeKernel([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6],[4, 16, 24, 16, 4],[1, 4, 6, 4, 1]], 1.0/256)
    sharpen5 = normalizeKernel([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4],[6, 24, -476, 24, 6],[4, 16, 24, 16, 4],[1, 4, 6, 4, 1]], -1.0/256)
    #blank_image = Convolution(blank_image, gaussianBlur5)
    #blank_image = Convolution(blank_image, sharpen5)
    cv2.imwrite('resukt.jpg', blank_image)
    print("Done")
    return
    

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
    global refPt, cropping, num
        
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(refPt) <4:
            refPt.append([x, y])
            draw([x, y])
            cropping = True
            print(refPt[num])
            num += 1
        else:
            imageProcess()
            num = 0
            refPt = []

            #sys.exit()
            
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
cv2.imshow("image", image)

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	key = cv2.waitKey(1) & 0xFF


# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	cv2.imshow("ROI", roi)
	cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()