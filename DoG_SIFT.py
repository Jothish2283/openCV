import cv2

img= cv2.imread("C:/Users/jothi/Pictures/Saved Pictures/Wallheaven/wallhaven-2819ly.png", 33)
img_f=img.copy()
cv2.imshow("original img", img)
cv2.waitKey(3000)
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift=cv2.xfeatures2d.SIFT_create()
keypoints, descriptors= sift.detectAndCompute(gray, None)
cv2.drawKeypoints(img, keypoints, img_f, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("descriptors img", img_f)
cv2.waitKey(5000)
cv2.destroyAllWindows()
for keypoint in keypoints:
    print("------keypoint------", keypoint.response)
    break