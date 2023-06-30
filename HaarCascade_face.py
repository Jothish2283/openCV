import cv2
img=cv2.imread("C:/Users/jothi/Downloads/fam.jpg", 33)
cv2.imshow("fam", img)
cv2.waitKey(3000)
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
haar_cascade_path="C:/Users/jothi/miniconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
face_detector=cv2.CascadeClassifier(haar_cascade_path)

faces=face_detector.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)
#scaleFactor-> the scale by which the image shd be decreased after every iteration
#minNeighbors-> the min amount of overlapping regions to confirm it is a face

for face in faces:
    x,y,w,h=face
    img_with_face_marked=cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    
cv2.imshow("marked_faces", img_with_face_marked)
cv2.waitKey(5000)
cv2.destroyAllWindows()