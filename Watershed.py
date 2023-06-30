import numpy as np
import cv2
from matplotlib import pyplot as plt

def img_loader(arr):
    arr= (arr-arr.min())/(arr.max()-arr.min())
    return arr

img = cv2.imread("C:/Users/jothi/Downloads/coins.jpg",33)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")
plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)


plt.imshow(img_loader(cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)))
plt.title("Threshold")
plt.axis("off")
plt.show()

# Remove noise.
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,iterations=2) #open op smoothens the boundaries.

plt.imshow(img_loader(cv2.cvtColor(opening,cv2.COLOR_GRAY2RGB)))
plt.title("Morph_Opening")
plt.axis("off")
plt.show()

# Find the sure background region.
sure_bg = cv2.dilate(opening, kernel, iterations=3)

plt.imshow(img_loader(cv2.cvtColor(sure_bg,cv2.COLOR_GRAY2RGB)))
plt.title("Sure_BG")
plt.axis("off")
plt.show()

# Find the sure foreground region.
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5) #calculates the distance between white and black pixels

plt.imshow(img_loader(cv2.cvtColor(dist_transform,cv2.COLOR_GRAY2RGB)))
plt.title("Dist_Image")
plt.axis("off")
plt.show()

ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = sure_fg.astype(np.uint8)

plt.imshow(img_loader(cv2.cvtColor(sure_fg,cv2.COLOR_GRAY2RGB)))
plt.title("Sure_FG")
plt.axis("off")
plt.show()

# Find the unknown region.
unknown = cv2.subtract(sure_bg, sure_fg)

plt.imshow(img_loader(cv2.cvtColor(unknown,cv2.COLOR_GRAY2RGB)))
plt.title("Unkown")
plt.axis("off")
plt.show()

# Label the foreground objects.
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1.
markers += 1

# Label the unknown region as 0.
markers[unknown==255] = 0

plt.imshow(img_loader(markers), cmap="tab20b")
plt.title("Markers")
plt.axis("off")
plt.show()

# Watershed initialization
markers = cv2.watershed(img, markers)
img[markers==-1] = [0, 255, 0] #segments are marked in green

plt.title("Seg_image")
plt.axis("off")
plt.imshow(img_loader(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
plt.show()