# Image processing

import cv2
from cvzone.HandTrackingModule import HandDetector #detects hand
import numpy as np
import math
import time
from cvzone.ClassificationModule import Classifier

classifier = Classifier("model\keras_model.h5","model\labels.txt")
labels =["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]
cap =cv2.VideoCapture(0)
hand_detector = HandDetector(maxHands=1)
data = "Dataset/A"

counter= 0

imgSize =300 #
offset=20 # for cropping image properly
while True:
    sucess, img = cap.read() 
    imgOutput =img.copy()
    hands,img  = hand_detector.findHands(img)
    if hands:
        hand =hands[0]
        x,y,w,h = hand['bbox']
        
        imgCrop =img[y-offset:y+h+offset,x-offset:x+w+offset] 
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255 #creating new matrix to fit captured image here for size uniformity
        
        # cv2.imshow("ImageCrop",imgCrop)
        # cv2.imshow("imgWhite",imgWhite)

        if h/w > 1: #if height is greater than width 
            k = imgSize/h #constant
            newWidth = math.ceil(k*w) #decresing the width
            imgResize = cv2.resize(imgCrop,(newWidth,imgSize)) #resizing cropped image to reduce variability 
            Wgap =math.ceil((imgSize-newWidth)/2)   #shifting imageWhite to center
            imgWhite[0:imgResize.shape[0],Wgap:Wgap+newWidth]= imgResize
        else :
            k = imgSize/w #constant
            newHeight = math.ceil(k*h) #decresing the width
            imgResize = cv2.resize(imgCrop,(imgSize,newHeight)) #resizing cropped image to reduce variability 
            hgap =math.ceil((imgSize-newHeight)/2)   #shifting imageWhite to center
            imgWhite[hgap:hgap+newHeight,0:imgResize.shape[1]]= imgResize
        prediction,index = classifier.getPrediction(imgWhite)    
        # cv2.imshow("ImageWhite",imgWhite) #final image 
        cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        cv2.imshow("ImageCopy",imgOutput)
        # print(prediction)
        # print( labels[index] )

    # cv2.imshow("Image",img)
    cv2.waitKey(1)
    