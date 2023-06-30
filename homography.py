import cv2
import matplotlib.pyplot as plt
import numpy as np

im0=cv2.imread("C:/Users/jothi/Downloads/query.png",0)
im1=cv2.imread("C:/Users/jothi/Downloads/anchor-man.png",0)

sift=cv2.xfeatures2d.SIFT_create()
kp0, des0=sift.detectAndCompute(im0, None)
kp1, des1=sift.detectAndCompute(im1, None)

flann_kd_index=1
index_params=dict(algorithm=flann_kd_index, trees=5)
search_params=dict(checks=50)

flann=cv2.FlannBasedMatcher(index_params, search_params)
pair_matches=flann.knnMatch(des0, des1, k=2)

thresh_matches= [match[0] for match in pair_matches if len(match)>0 and match[0].distance <0.7*match[1].distance]
good_matches=sorted(thresh_matches, key= lambda x: x.distance)[:10]

src_pts=np.float32([kp0[m.queryIdx].pt for m in good_matches])#.reshape(-1,1,2) #-1 indicates current shape
dest_pts=np.float32([kp1[m.trainIdx].pt for m in good_matches])#.reshape(-1,1,2)

T, mask=cv2.findHomography(src_pts, dest_pts, cv2.RANSAC, 5.0) #5->ransac threshold
mask_matches=mask.ravel().tolist()

# =============================================================================
# Finding the detection box
# =============================================================================

w,h=im0.shape
src_corners= np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
dest_corners=cv2.perspectiveTransform(src_corners, T) #T is the transformation matrix
dest_corners=dest_corners.squeeze().astype(np.int32) #non-conversion to integral value leads to error

for i in range(len(dest_corners)-1):
    x0,y0=dest_corners[i]
    x1,y1=dest_corners[i+1]
    cv2.line(im1, (x0,y0), (x1,y1), [0,0,255], 3)
    if i==2:
        x0,y0=dest_corners[3]
        x1,y1=dest_corners[0]
        cv2.line(im1, (x0,y0), (x1,y1), [0,0,255], 3)
        
img_match=cv2.drawMatches(im0, kp0, im1, kp1, good_matches, im1,
                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        
plt.imshow(img_match)