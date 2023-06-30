import cv2
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np

encoder=LabelEncoder()

def read_img(dir, img_size):
    idx=0
    imgs,labels=[],[]
    for dir_name, sub_dirname, files in os.walk(dir):
        if sub_dirname:
            sub_dirname_l=sub_dirname
            codes=encoder.fit_transform(list(sub_dirname))
            class_dict={sub_dirname[i]:codes[i] for i in range(len(sub_dirname))}
            
        if files:
            for file in files:
                path=dir+f"/{sub_dirname_l[idx]}/"+file
                img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img=cv2.resize(img, (img_size, img_size))
                imgs.append(img)
                labels.append(int(class_dict[sub_dirname_l[idx]]))
            idx+=1
    imgs = np.asarray(imgs, np.uint8)
    labels = np.asarray(labels, np.int32)
    return imgs, labels, class_dict

img_size=200
train_imgs, train_labels, class_dict=read_img("C:/Users/jothi/Downloads/Face", img_size)

model=cv2.face.EigenFaceRecognizer_create(num_components=100, threshold=8000)
model.train(train_imgs, np.array(train_labels))

face_path="C:/Users/jothi/miniconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
face_detector=cv2.CascadeClassifier(face_path)
vid_c=cv2.VideoCapture(0)
success, frame=vid_c.read()

while success:
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(gray, scaleFactor=1.025, minNeighbors=20, minSize=(120,120))
    if faces !=():
        for face in faces:
            x,y,w,h=face
            test_img=gray[y:y+h, x:x+w]
            test_img=cv2.resize(test_img, (img_size, img_size))
            pred_label, pred_prob=model.predict(test_img)
            text=f"pred_class: {[name for name in class_dict if class_dict[name]==pred_label][0]} pred_prob: {pred_prob :.2f}"
            cv2.putText(frame, text,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.imshow("detected_face", frame)
            cv2.waitKey(1)
    else:
        cv2.imshow("detected_face", frame)
        cv2.waitKey(1)
        
    success, frame=vid_c.read()
    if cv2.waitKey(1)==27:
        cv2.destroyAllWindows()
        vid_c.release()
        break
        