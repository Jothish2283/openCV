import cv2
import matplotlib.pyplot as plt

im_logo=cv2.imread("C:/Users/jothi/Downloads/NASA_logo.jpg", 32)
im_scene=cv2.imread("C:/Users/jothi/Downloads/Kennedy_space_center.jpg",16)

orb=cv2.ORB_create()
kp0, desc0=orb.detectAndCompute(im_logo, None)
kp1, desc1=orb.detectAndCompute(im_scene, None)

#Brute Force matching
bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches= bf.match(desc0, desc1)

matches_sorted=sorted(matches, key= lambda x: x.distance)
img_match=cv2.drawMatches(im_logo, kp0, im_scene, kp1, matches_sorted[:25], #only taking top 25 ie 25 least distance
                          im_scene,
                          flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img_match)
plt.title("Brute_force_normal")
plt.show()

# =============================================================================
# Using KNN matcher
# =============================================================================

bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) #when knn is used cross check is set False
pair_matches=bf.knnMatch(desc0, desc1, k=2) #returns a list of 2 keypt whose dist are sorted ascedingly

pair_matches_sorted=sorted(pair_matches, key=lambda x: x[0].distance) #sorts the outer most list

#set threshold
best_matches=[x[0] for x in pair_matches_sorted if len(x)>0 and x[0].distance < 0.8*x[1].distance]

img_match=cv2.drawMatches(im_logo, kp0, im_scene, kp1, best_matches[:25], 
                          im_scene,
                          flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img_match)
plt.title("Brute_force_knn")
plt.show()