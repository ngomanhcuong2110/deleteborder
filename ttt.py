"""
import cv2
import numpy as np
import random as rng
import matplotlib.pyplot as plt
from numpy.lib.type_check import imag
kernel = np.ones((3,3),np.uint8)
kernel1 = np.ones((7,7),np.uint8)
import sys,os
import glob
sys.path.append(os.path.realpath('..'))
list=glob.glob(sys.path[0]+"//*png")
save=sys.path[0]+"//te//"
import time
import imutils
start=time.time()
i=0
def create_jpg(im):
# Separate BGR channels from A, make everything float in range 0..1
    BGR = im[...,0:3].astype(np.float)/255
    A   = im[...,3].astype(np.float)/255
    # First, composite image over black background using:
    # result = alpha * Foreground + (1-alpha)*Background
    # Now, composite image over white background
    bg  = np.zeros_like(BGR).astype(np.float)+1   # white background
    fg  = A[...,np.newaxis]*BGR                   # new alpha-scaled foreground
    bg = (1-A[...,np.newaxis])*bg                 # new alpha-scaled background
    res = cv2.add(fg, bg)                         # sum of the parts
    res = (res*255).astype(np.uint8)              # scaled back up
    return res
def is_contour_bad(c):
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# the contour is 'bad' if it is not a rectangle
	return not len(approx) == 4

imag = cv2.imread("4.png")

mask1 = np.ones(imag.shape, dtype="uint8") * 255
image=imag+mask1

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 250, 120)
raw=cv2.bitwise_and(image,image,mask1)

_,thr=cv2.threshold(raw,127,255,cv2.THRESH_BINARY)
cv2.imshow("Original", image)
# find contours in the image and initialize the mask that will be
# used to remove the bad contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
mask = np.ones(image.shape[:2], dtype="uint8") * 255
cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x),reverse=True)
i=0
# loop over the contours
for c in cntsSorted: 

    if is_contour_bad(c):
        cv2.drawContours(mask, [c], -1, (i,i,i), -1)
    

# remove the contours from the image and show the resulting images
image = cv2.bitwise_and(image, image, mask=mask)
cv2.drawContours(raw, [cntsSorted[4]], -1, (255,255,255), 12)
cv2.imshow("raw",raw)
#cv2.imshow("After", image)
cv2.waitKey(0)
"""
"""
import cv2
import numpy as np
def create_jpg(im):
# Separate BGR channels from A, make everything float in range 0..1
    BGR = im[...,0:3].astype(np.float)/255
    A   = im[...,3].astype(np.float)/255
    # First, composite image over black background using:
    # result = alpha * Foreground + (1-alpha)*Background
    # Now, composite image over white background
    bg  = np.zeros_like(BGR).astype(np.float)+1   # white background
    fg  = A[...,np.newaxis]*BGR                   # new alpha-scaled foreground
    bg = (1-A[...,np.newaxis])*bg                 # new alpha-scaled background
    res = cv2.add(fg, bg)                         # sum of the parts
    res = (res*255).astype(np.uint8)              # scaled back up
    return res
# load image
img = cv2.imread('7.png',cv2.IMREAD_UNCHANGED)

mask1 = np.ones(img.shape, dtype="uint8") * 255
re=img+mask1
np.where(re>255,re,re-255)
re=np.asarray(re,dtype=np.uint8)
print(np.max(re))

im_gray=cv2.cvtColor(re,cv2.COLOR_BGR2GRAY)
_,thr=cv2.threshold(im_gray,127,255,cv2.THRESH_BINARY)
img=create_jpg(img)
thr=cv2.bitwise_and(img,img,thr)

cv2.imshow("vbv",thr)
edged = cv2.Canny(im_gray, 120, 255)

contours, hierarchy = cv2.findContours(edged, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  
cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey(0)
  
print("Number of Contours found = " + str(len(contours)))
  
# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(re, contours, -1, (0,0,0), thickness=cv2.FILLED)
# convert to graky



cv2.imshow("White",re)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
import imutils

import cv2
from PIL import Image

import time
start=time.time()
import matplotlib.pyplot as plt
import numpy as np
import sys,os
import glob
sys.path.append(os.path.realpath('..'))
list=glob.glob(sys.path[0]+"//*png")
save=sys.path[0]+"//te//"
rgba_image = Image.open("1.png")
if (rgba_image.mode=="LA"):
    rgba_image=rgba_image.convert("RGBA")

rgba_image.load()
background = Image.new("RGB", rgba_image.size, (255, 255, 255))
background.paste(rgba_image, mask = rgba_image.split()[3])
#background.save("sample_2.jpg", "JPEG", quality=100)
backgrounds=np.asarray(background,dtype=np.uint8)

background=cv2.cvtColor(backgrounds,cv2.COLOR_RGB2BGR)
gray=cv2.cvtColor(background,cv2.COLOR_BGR2GRAY)
edge=cv2.Canny(gray,10,255)
contours, hierarchy = cv2.findContours(edge, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)



# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(background, contours, -1, (0,0,0), thickness=cv2.FILLED)
background=cv2.cvtColor(background,cv2.COLOR_BGR2GRAY)
#cv2.imshow("contour",background)

kernel = np.ones((5,5),np.uint8)
kernel1 = np.ones((9,9),np.uint8)
erosion = cv2.erode(background,kernel,iterations = 1)
dilate=cv2.dilate(background,kernel,iterations = 1)
sub=dilate-erosion
subnot=cv2.bitwise_not(sub)
cv2.imshow("subnot",subnot)

im=cv2.bitwise_and(backgrounds,backgrounds,mask=sub)
im=cv2.bitwise_not(im)
r,thr=cv2.threshold(im,70,255,cv2.THRESH_BINARY_INV)
cv2.imshow("iuiui",thr)
i_last=np.copy(backgrounds)
i_last.setflags(write=1)
thrr=cv2.cvtColor(thr,cv2.COLOR_BGR2GRAY)
conts, h = cv2.findContours(thrr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(backgrounds, [conts[0]], -1, (255,255,255), 20)
cv2.imshow("thr",backgrounds)


"""
for i in range (0,i_last.shape[0]):
    for j in range(0,i_last.shape[1]):
        
        
        comparer=thr[i][j]==[255, 255, 255]
    
        if(comparer.all()):
            i_last[i,j]=(255,255,255)


i_last=cv2.cvtColor(i_last,cv2.COLOR_RGB2BGR)
"""
cv2.imwrite("hh.jpg",i_last)
print(time.time()-start)

cv2.waitKey(0)