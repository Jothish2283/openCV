import cv2
import matplotlib.pyplot as plt

im0= cv2.imread("C:/Users/jothi/Pictures/Saved Pictures/peakpx.jpg",32)
im1=cv2.imread("C:/Users/jothi/Downloads/img_cluster.png", 16)

sift = cv2.xfeatures2d.SIFT_create()
kp0, des0 = sift.detectAndCompute(im0, None)
kp1, des1 = sift.detectAndCompute(im1, None)

flann_index_kdtree=1
index_params=dict(algorithm= flann_index_kdtree, trees=5)
search_params= dict(checks=50)

flann= cv2.FlannBasedMatcher(index_params, search_params)
pair_matches=flann.knnMatch(des0, des1, k=2)
pair_matches_sorted=sorted(pair_matches, key=lambda x: x[0].distance) #sorts the outer most list

#set threshold
best_matches=[x[0] for x in pair_matches_sorted if len(x)>0 and x[0].distance < 0.7*x[1].distance]

img_match=cv2.drawMatches(im0, kp0, im1, kp1, best_matches[:25], 
                          im1,
                          flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img_match)
plt.title("Flann_knn")
plt.show()