import cv2
import os

maskPath = os.path.abspath(r'C:\Users\timhr\projects\python\maskDetetection\dataMoreNeg\cascade.xml')
maskCasc = cv2.CascadeClassifier(maskPath)

facePath = os.path.abspath(r'C:\Users\timhr\Documents\opencv\build\etc\haarcascades\haarcascade_frontalface_alt.xml')
faceCasc = cv2.CascadeClassifier(facePath)

cv2.namedWindow("Mask Detection")
vc = cv2.VideoCapture(0)

frameNum = 0

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    print("rval true")
else:
    rval = False
    print("rval false")

while rval:
    rval, frame = vc.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    masks = maskCasc.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=16, minSize=(200, 200), flags=cv2.CASCADE_SCALE_IMAGE)
    facesTemp = faceCasc.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=1, minSize=(200, 200), flags=cv2.CASCADE_SCALE_IMAGE)
    
    if(frameNum == 0):
        faces = facesTemp
    
    if(len(facesTemp)>0):
        faces = facesTemp
    
    for(xFace,yFace,wFace,hFace) in faces:
        if(len(masks)>0):
            cv2.rectangle(frame, (xFace,yFace),(xFace+wFace,yFace+hFace),(0,255,0), 2)
            cv2.putText(frame, 'Thank you for wearing a mask', (xFace, yFace-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        else:
            cv2.rectangle(frame, (xFace,yFace),(xFace+wFace,yFace+hFace),(0,0,255), 2)
            cv2.putText(frame, 'Please put on a mask', (xFace, yFace-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    cv2.imshow("Mask Detection", frame)
    
    frameNum+=1
    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    
vc.release()
cv2.destroyAllWindows()