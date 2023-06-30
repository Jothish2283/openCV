import cv2
import numpy as np

img=cv2.imread("C:/Users/jothi/Pictures/Saved Pictures/1007550.jpg", cv2.IMREAD_REDUCED_COLOR_4)
cv2.imshow("img",img)
img_g=cv2.imread("C:/Users/jothi/Pictures/Saved Pictures/1007550.jpg", cv2.IMREAD_REDUCED_GRAYSCALE_4)
cv2.imshow("img_g",img_g)
cv2.waitKey(3*1000) #3*1000 millisec is the life time of the window;
#if 0 is passed the image lives till it is closed manually 

print("img shape: ",np.array(img).shape) #(271,505,3)

#split and merge channels
b,g,r= cv2.split(img)
print("b/g/r_channel shape: ",b.shape)
cv2.imshow("b",b)
cv2.waitKey(3000)
cv2.imshow("g",g)
cv2.waitKey(3000)
cv2.imshow("r",r)
cv2.waitKey(3000)

img_rgb=cv2.merge([r,g,b])
cv2.imshow("img_merged_RGB", img_rgb)
cv2.waitKey(3000)

img_gbr=cv2.merge([g,b,r])
cv2.imshow("img_merged_GBR", img_gbr)
cv2.waitKey(3000)

img_c=img.copy()
img_c[200:,450:]=[0,255,0] #converts pixels after (200, 450) to GREEN
#CV2 follows BGR

cv2.imshow("img_edit",img_c)
cv2.waitKey(3*1000)

cv2.destroyAllWindows()