import cv2
import numpy as np
import os



def distance(v1 , v2):
    return np.sqrt(((v1-v2)**2).sum())

def knn(train , test , k=5):
    dist=[]
    for i in range(train.shape[0]):
        ix = train[i , :-1]
        iy = train[i, -1]
        d = distance(test , ix)
        dist.append([d , iy])
    dk = sorted(dist , key=lambda x:x[0])[:k]
    labels = np.array(dk)[: , -1]
    output = np.unique(labels , return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]

cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
skip=0
facedata=[]
label=[]
classid=0
names={}

for fx in os.listdir("Datasets"):
    if fx.endswith('.npy'):
        names[classid] = fx[:-4]
        print("loaded : "+fx)
        dataitem = np.load("Datasets/"+fx)
        facedata.append(dataitem)
        target = classid*np.ones((dataitem.shape[0] , ))
        classid+=1
        label.append(target)

totalfacedata = np.concatenate(facedata , axis=0)
totallabel = np.concatenate(label , axis=0).reshape((-1 , 1))

trainset = np.concatenate((totalfacedata , totallabel) , axis=1)

while True:
    success , frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray , 1.3 , 5)
    for face in faces[-1:]:
        x, y, w, h = face
        offset = 10
        facesection = frame[y - offset:y + h + offset, x - offset: x + w + offset]
        facesection = cv2.resize(facesection, (100, 100))
        out = knn(trainset , facesection.flatten())
        predname = names[int(out)]
        cv2.putText(frame , predname , (x,y-10) , cv2.FONT_HERSHEY_COMPLEX , 1 , (255 , 0 , 0) , 2 , cv2.LINE_AA)
        cv2.rectangle(frame , (x,y) , (x+w , y+h) , (255,0,0) , 2)
    cv2.imshow("Faces" , frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break