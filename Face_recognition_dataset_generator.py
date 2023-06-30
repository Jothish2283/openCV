import cv2
from pathlib import Path

haar_cascade_pathf="C:/Users/jothi/miniconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
face_detector=cv2.CascadeClassifier(haar_cascade_pathf)

user= input("Name of user:\n")
saving_path="C:/Users/jothi/Downloads/Face/"+user
print("saving user photos to:\n",saving_path)
Path(saving_path).mkdir(parents=True, exist_ok=True)
idx=-40 #take care of start delay and get stable imgs

vid_c=cv2.VideoCapture(0)
success, frame=vid_c.read()

while success:
    marked_faces=frame.copy()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(gray, scaleFactor=1.025, minNeighbors=20, minSize=(120,120))
    for face in faces:
        x,y,w,h=face
        marked_faces=cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        save_face=cv2.resize(gray[y:y+h, x:x+w], (200,200))
        
    if idx>=0: 
        file_name="/"+str(idx)
        cv2.imwrite(f"{saving_path+file_name}.jpg", save_face)
    cv2.imshow("output", marked_faces)
    cv2.waitKey(1)
    success, frame=vid_c.read()
    idx+=1
    if cv2.waitKey(1)==27: #press escape to close
        cv2.destroyAllWindows()
        vid_c.release()
        break
