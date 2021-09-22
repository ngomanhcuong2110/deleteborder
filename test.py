
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
save=sys.path[0]+"//save//"
for imag in list:
    rgba_image = Image.open(imag)
    if (rgba_image.mode=="LA"):
        rgba_image=rgba_image.convert("RGBA")

    rgba_image.load()
    background = Image.new("RGB", rgba_image.size, (255, 255, 255))
    background.paste(rgba_image, mask = rgba_image.split()[3])
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

    kernel = np.ones((3,3),np.uint8)
    kernel1 = np.ones((9,9),np.uint8)
    erosion = cv2.erode(background,kernel,iterations = 1)
    dilate=cv2.dilate(background,kernel,iterations = 1)
    sub=dilate-erosion
    subnot=cv2.bitwise_not(sub)

    im=cv2.bitwise_and(backgrounds,backgrounds,mask=sub)
    im=cv2.bitwise_not(im)
    r,thr=cv2.threshold(im,70,255,cv2.THRESH_BINARY_INV)
    i_last=np.copy(backgrounds)
    i_last.setflags(write=1)
    thrr=cv2.cvtColor(thr,cv2.COLOR_BGR2GRAY)
    conts, h = cv2.findContours(thrr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(i_last, [conts[0]], -1, (255,255,255), 20)

    """
    
    """
    i_last=cv2.cvtColor(i_last,cv2.COLOR_RGB2BGR)
    print(imag.split("\\")[-1])
    cv2.imwrite(save+imag.split("\\")[-1][:-4]+".jpg",i_last)
print(time.time()-start)

cv2.waitKey(0)