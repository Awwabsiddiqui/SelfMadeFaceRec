import cv2
import numpy as np


cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
skip=0
facedata=[]
filename = input("enter the name of persoon : ")
while True:
    success , frame = cap.read()
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray , 1.3 , 5)
    faces = sorted(faces , key=lambda f:f[2]*f[3])
    for face in faces[-1:]:
        x, y, w, h = face
        cv2.rectangle(frame , (x,y) , (x+w , y+h) , (255,0,0) , 2)
        offset=10
        facesection = frame[y-offset:y+h+offset , x-offset : x+w+offset]
        facesection = cv2.resize(facesection , (100 , 100))
        skip+=1
        if skip%10==0:
            facedata.append(facesection)
            #print(len(facedata))
    #cv2.imshow("frame", frame)
    cv2.imshow("facesection", facesection)

    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
facedata = np.asarray(facedata)
facedata = facedata.reshape((facedata.shape[0] , -1))
print(facedata.shape)
np.save("Datasets/"+filename + ".npy" , facedata)
cap.release()
cv2.destroyAllWindows()