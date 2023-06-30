import cv2
import numpy as np
import glob
from sklearn import svm
from joblib import dump, load
from non_max_suppression import non_max_suppression_fast as nms

BOW_NUM_SAMPLES=490
SVM_NUM_SAMPLES=100
train_dir="C:/Users/jothi/Downloads/CarData/CarData/TrainImages"
test_dir="C:/Users/jothi/Downloads/CarData/CarData/TestImages"

def get_train_samples(num_samples, label, work_dir=train_dir):
    pos_paths=glob.glob(f"{work_dir}/pos-*.pgm")
    neg_paths=glob.glob(f"{work_dir}/neg-*.pgm")
    if label=="pos": paths=pos_paths
    else: paths=neg_paths
    
    return paths[:num_samples]
    
sift= cv2.SIFT_create()
flann_index_tree=1
index_params=dict(algorithm=flann_index_tree, trees=5)
search_params=dict(checks=50)
flann=cv2.FlannBasedMatcher(index_params, search_params)

bow_kmeans_trainer=cv2.BOWKMeansTrainer(50) #50->number of clusters
bow_extractor=cv2.BOWImgDescriptorExtractor(sift, flann)

def train_bow(num_samples=BOW_NUM_SAMPLES):
    pos_paths=get_train_samples(num_samples, "pos")
    for path in pos_paths:
        img=cv2.imread(path, 0)
        kp, desc=sift.detectAndCompute(img, None)
        if desc is not None: bow_kmeans_trainer.add(desc)
        
    neg_paths=get_train_samples(num_samples, "neg")
    for path in neg_paths:
        img=cv2.imread(path, 0)
        kp, desc=sift.detectAndCompute(img, None)
        if desc is not None: bow_kmeans_trainer.add(desc)
        
train_bow(BOW_NUM_SAMPLES)

voc=bow_kmeans_trainer.cluster()
bow_extractor.setVocabulary(voc)
# now we can use bow extractor to directly get bow_descriptors from DOG used by sift,
# bow descriptors are higher level than normal sift descriptors.

def extract_bow_feature(img):
    features=sift.detect(img)
    return bow_extractor.compute(img, features)

def data_svm(num_samples=SVM_NUM_SAMPLES):
    training_samples, training_labels=[],[]
    pos_paths=get_train_samples(num_samples, "pos")
    for path in pos_paths:
        img=cv2.imread(path, 0)
        bow_desc=extract_bow_feature(img)
        if bow_desc is not None:
            training_samples.extend(bow_desc)
            training_labels.append(1)
        
    neg_paths=get_train_samples(num_samples, "neg")
    for path in neg_paths:
        img=cv2.imread(path, 0)
        bow_desc=extract_bow_feature(img)
        if bow_desc is not None:
            training_samples.extend(bow_desc)
            training_labels.append(0)
    
    return np.array(training_samples), np.array(training_labels)

train_data, train_labels=data_svm(SVM_NUM_SAMPLES)

# print("\n\n------train_data-------", train_data)
# print("\n\n------shape--------", train_data.shape)
# print("\n\n-------label_shape-------", train_labels.shape)

clf=svm.SVC(probability=True)
clf.fit(train_data, train_labels)
print("classifier score", clf.score(train_data, train_labels)) 

predict_img_path=['C:/Users/jothi/Downloads/CarData/CarData/TestImages/test-0.pgm',
                  'C:/Users/jothi/Downloads/CarData/CarData/TestImages/test-1.pgm',
                  'C:/Users/jothi/Downloads/CarData/CarData/TestImages/test-5.pgm',
                  'C:/Users/jothi/Downloads/CarData/CarData/TestImages/test-12.pgm',
                  'C:/Users/jothi/Downloads/CarData/CarData/TestImages/test-15.pgm',
                  'C:/Users/jothi/Downloads/CarData/CarData/TestImages/test-17.pgm',
                  'C:/Users/jothi/Downloads/CarData/CarData/TestImages/test-21.pgm']
                  
# =============================================================================
# Predicting boxes
# =============================================================================

def img_pyramid(img, scale_factor=1.25, min_size=(200, 80), max_size=(600,600)):
    h,w=img.shape
    w_min,h_min=min_size
    w_max, h_max=max_size
    
    while w>=w_min and h>=h_min:
        if w<=w_max and h<=h_max:
            yield img #yield is used like return but for generators
        
        w/=scale_factor
        h/=scale_factor
        img=cv2.resize(img, (int(w), int(h)),
                       interpolation=cv2.INTER_AREA)
        
def sliding_window(img, step=10, window_size=(100,40)):
    h,w=img.shape
    w_win, h_win=window_size
    for y in range(0, w, step):
        for x in range(0, h, step):
            roi=img[y:y+h_win, x:x+w_win]
            h_roi, w_roi=roi.shape
            if w_roi==w_win and h_roi==h_win:
                yield (x, y, roi)


def predict(img_paths=predict_img_path, nms_threshold=0.25):
    boxes=[]
    for path in img_paths:
        img_o=cv2.imread(path,1)
        img=cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY)
        for scaled_img in img_pyramid(img):
            for x,y,roi in sliding_window(scaled_img):
                bow_desc=extract_bow_feature(roi)
                if bow_desc is None: continue
                pred=clf.predict(bow_desc)[0]
                pred_c=clf.predict_proba(bow_desc)[0][1]
                if pred==1: #car
                    h,w=roi.shape
                    scale=img.shape[0]/roi.shape[0]
                    x,y,X,Y=x*scale, y*scale, (x+w)*scale, (y+h)*scale
                    boxes.append([int(x), int(y), int(X), int(Y), pred_c])
        
        true_boxes=nms(np.array(boxes), nms_threshold)
        for x,y,X,Y,p in true_boxes:
            text=f"car: {p :.3f}"
            cv2.rectangle(img_o, (int(x),int(y)), (int(X),int(Y)), (0,255,0), 2)
            cv2.putText(img_o, text, (int(x),int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,255,255], 2, cv2.LINE_AA)
        cv2.imshow("pred", img_o)
        cv2.waitKey(-1)
    cv2.destroyAllWindows()

predict(predict_img_path)
save_path="car_detection.joblib"
print(f"saving_model to: {save_path}")
dump(clf, save_path)
clf_loaded=load(save_path)