import cv2
import numpy as np

# img=cv2.imread("C:/Users/jothi/Pictures/Saved Pictures/Wallheaven/wallhaven-odq7o5.png", 33)
# img_g=cv2.imread("C:/Users/jothi/Pictures/Saved Pictures/Wallheaven/wallhaven-odq7o5.png", 32)

img=cv2.imread("C:/Users/jothi/Downloads/WhatsApp Image 2023-05-30 at 21.51.53.jpeg", 33)
img_g=cv2.imread("C:/Users/jothi/Downloads/WhatsApp Image 2023-05-30 at 21.51.53.jpeg", 32)

img_blur= cv2.GaussianBlur(img_g, (3,3), 0) #for denoising
                         
cv2.imshow("img_blur", img_blur)
cv2.waitKey(3000)

img_edit3=img_blur.copy()
#-1 in depth ie in place of cv2.CV_8U means same dept as input img
cv2.Laplacian(img_blur, cv2.CV_8U, img_edit3, 5) #3 are kernel sizes
# w=(255-img_edit3)/255. #inversion: white colour edges-> black colour edges
# channels=cv2.split(img)

# for channel in channels:
#     channel[:]=w*channel

# img_edit3=cv2.merge(channels)
cv2.imshow("img_Laplacian", img_edit3)
cv2.waitKey(3000)

_,thresh_img=cv2.threshold(img_edit3, 250, 255, 0)
cv2.imshow("thresh_img", thresh_img)
cv2.waitKey(3000)

img_canny=cv2.Canny(img_g, 200, 100) #200->1st threshold; 300->2nd threshold
cv2.imshow("img_Canny", img_canny)
cv2.waitKey(3000)
# =============================================================================
# HoughLine detection
# =============================================================================
rho, theta=1, np.pi/180 #positional & rotational step size
threshold, min_line_lenght, max_line_gap= 2, 1, 0 #voting threshold
lines=np.squeeze(cv2.HoughLinesP(img_canny, rho, theta, threshold, min_line_lenght, max_line_gap))
for line in lines:
    x1,y1,x2,y2=line
    cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 1)

cv2.imshow("img_detectedLines", img)
cv2.waitKey(3000)
# =============================================================================
# HoughCircle detection
# =============================================================================
img=cv2.imread("C:/Users/jothi/Downloads/coins.jpg", 33)
cv2.imshow("coin_img", img)
cv2.waitKey(3000)

img_g=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("coin_img_g", img_g)
cv2.waitKey(3000)

dp, min_dist=1, 120
param1, param2= 350, 140
min_rad, max_rad=0, 0

circles=np.squeeze(cv2.HoughCircles(img_g, cv2.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2,
                         minRadius=min_rad, maxRadius=max_rad))

circles=np.int0(circles)

for cir in circles:
    x,y,r=cir
    cv2.circle(img, (x,y), r, (0,255,0), 2) #circle
    cv2.circle(img, (x,y), 2, (0,0,255), 2) #center
    
cv2.imshow("detected_circles", img)
cv2.waitKey(4000)
cv2.destroyAllWindows()