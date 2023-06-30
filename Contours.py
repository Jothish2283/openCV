import cv2
import numpy as np

# =============================================================================
# Simple artificial example
# =============================================================================
img = np.zeros((200, 200), dtype=np.uint8)
img[50:150, 50:150] = 255

cv2.imshow("img", img)
cv2.waitKey(3000)

ret, thresh_img = cv2.threshold(img, 125, 255, 0) #0==THRESH_BINARY
#255 is max value ie f(x)=0 if x<thresh[125]; f(x)=255 if x>thresh

#print(ret, thresh_img[50:150, 50:150]) #ret gives thresh val
cv2.imshow("thresh_img", thresh_img)
cv2.waitKey(3000)

contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #cv2.RETR_TREE-gives the contours in hierarchial form
#cv2.CHAIN_APPROX_SIMPLE- gives the end points loc

#print(contours, hierarchy)

color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = cv2.drawContours(color, contours, -1, (0,255,0), 2) #-1 or any -ve number indicates to mark all contors, 2 gives thickness -ve number fills the space with colour
cv2.imshow("contours", color)
cv2.waitKey(3000)

# =============================================================================
# More realistic/complicated example
# =============================================================================
img=cv2.imread("C:/Users/jothi/Downloads/WhatsApp Image 2023-05-30 at 21.51.53.jpeg",33)
cv2.imshow("img", img)
cv2.waitKey(3000)

ret, thresh_img=cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 255, 0)
cv2.imshow("thresh_img", thresh_img)
cv2.waitKey(3000)

c, h=cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

img = cv2.drawContours(img, c, -1, (0,255,0), 2)
cv2.imshow("contours", img)
cv2.waitKey(3000)
cv2.destroyAllWindows()