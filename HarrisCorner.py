import cv2

img=cv2.imread("C:/Users/jothi/Downloads/geo_shapes.jpg", 1)
cv2.imshow("original", img)
cv2.waitKey(3000)
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dist=cv2.cornerHarris(gray, 5,3,0.001) #3rd parameter is most imp->kernel size x(-[3,31] 3-> v.sensitive
img[dist>0.01*dist.max()]=[0,0,225]
cv2.imshow("with corners marked", img)
cv2.waitKey(5000)
cv2.destroyAllWindows()