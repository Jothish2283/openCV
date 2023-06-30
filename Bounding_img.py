import cv2
import numpy as np

img=cv2.imread("C:/Users/jothi/Downloads/thunder_bolt.PNG", 1)
cv2.imshow("original", img)
cv2.waitKey(3000)

ret, thresh_img=cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 220, 255, 0)
cv2.imshow("thresh_image", thresh_img)
cv2.waitKey(3000)

C, h= cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img_contour=cv2.drawContours(img, C,-1, (0,255,0), 1)
cv2.imshow("contours", img_contour)
cv2.waitKey(3000)

for c in C:
    x,y,w,h=cv2.boundingRect(c)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
    
    rect=cv2.minAreaRect(c) #returns x,y,w,h,and angle of rotation
    box=cv2.boxPoints(rect) #to compute the coordinates of edges
    box=np.int0(box) #to remove -ve value and normalize to int
    cv2.drawContours(img, [box], 0, (0,0,255), 2)
    
    (x,y), r= cv2.minEnclosingCircle(c)
    centre, r=(int(x), int(y)), int(r)
    cv2.circle(img, centre, r, (0,255,0), 2)
    
cv2.imshow("bounded_image", img)
cv2.waitKey(3000)

black=np.zeros_like(img)

for c in C:
    epsilon= 0.01*cv2.arcLength(c, True) #boolean: is the contour closed; 0.01-> 1% permissable error in polygon perimeter
    poly=cv2.approxPolyDP(c, epsilon, True)         
    hull=cv2.convexHull(c)
    
    cv2.drawContours(black, [c], 0, (0,255,0), 1)
    cv2.drawContours(black, [poly], 0, (255,255,0), 2)
    cv2.drawContours(black, [hull], 0, (0,0,255), 2)
    
cv2.imshow("poly_bound_image", black)
cv2.waitKey(3000)
cv2.destroyAllWindows()