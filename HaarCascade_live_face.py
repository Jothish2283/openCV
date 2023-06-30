import cv2

haar_cascade_pathf="C:/Users/jothi/miniconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
haar_cascade_pathe="C:/Users/jothi/miniconda3/Lib/site-packages/cv2/data/haarcascade_eye.xml"
face_detector=cv2.CascadeClassifier(haar_cascade_pathf)
eye_detector=cv2.CascadeClassifier(haar_cascade_pathe)
vid_c=cv2.VideoCapture(0)
success, frame=vid_c.read()

while success:
    marked_faces=frame.copy()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(gray, scaleFactor=1.025, minNeighbors=3, minSize=(120,120))
    for face in faces:
        x,y,w,h=face
        marked_faces=cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        roi=gray[y:y+h, x:x+w] #pixel val are (H,W) ie height first width second
        eyes=eye_detector.detectMultiScale(gray, scaleFactor=1.025, minNeighbors=5, minSize=(40,40))
        for eye in eyes:
            x,y,w,h=eye
            marked_faces=cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
        
    cv2.imshow("output", marked_faces)
    cv2.waitKey(1)
    success, frame=vid_c.read()
    if cv2.waitKey(1)==27: #press escape to close
        cv2.destroyAllWindows()
        vid_c.release()
        break
