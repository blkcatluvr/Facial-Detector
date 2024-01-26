import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #reads face

def detect_faces(img): #pass in img
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #facial detection works best on gray scale images
    faces = face_classifier.detectMultiScale(gray,1.3,5) #using face train classifier
    if faces is():
        return img
    
    for (x,y,w,h) in faces: #looping through face array and coordinates the w and h of each face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw the img with rectanges 
    return img 

cap = cv2.VideoCapture(0) #display continuously in real time.
while True:
    ret, frame = cap.read()
    frame = detect_faces(frame)
    cv2.imshow('Video Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
