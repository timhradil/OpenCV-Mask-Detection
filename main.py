"""
    Inputs webcam video and determines if user is wearing a mask or not.

    @author Tim Hradil
"""
import cv2
import os

maskPath = os.path.abspath(r'C:\Users\timhr\projects\python\maskDetetection\dataMoreNeg\cascade.xml')
maskCasc = cv2.CascadeClassifier(maskPath)

cv2.namedWindow('Mask Detection', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Mask Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

vc = cv2.VideoCapture(0)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frameNum = 0

maskScale = -20
mask = False

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    print("rval true")
else:
    rval = False
    print("rval false")

while rval:
    rval, frame = vc.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[100:650,350:1000]
    
    masks = maskCasc.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=950, minSize=(250, 250), flags=cv2.CASCADE_SCALE_IMAGE)
    
    if(len(masks)>0):
        maskScale+=1
        if(maskScale>=10):
            mask = True
            if(maskScale>=20):
                maskScale = 20
    else:
        maskScale-=1
        if(maskScale<=-10):
            mask = False
            if(maskScale<=-20):
                maskScale = -20
                
    if(mask):
        cv2.rectangle(frame, (450,150),(800,600),(0,255,0), 2)
        cv2.putText(frame, 'Thank you for wearing a mask', (500, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    else:
        cv2.rectangle(frame, (450,150),(800,600),(0,0,255), 2)
        cv2.putText(frame, 'Position face in this box and put on mask', (400, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    
    cv2.imshow("Mask Detection", frame)
    
    frameNum+=1
    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    
vc.release()
cv2.destroyAllWindows()