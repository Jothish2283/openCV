import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

original = cv2.imread("C:/Users/jothi/Downloads/cat_foreground.jpg")
img = original.copy()
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (373, 380, 823, 699) #values from ybat
mask, bgdmodel, fgdmodel=cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
img_without_mask = img*mask2[:,:,np.newaxis] #broadcasting

img_for_masking=cv2.cvtColor(img_without_mask, cv2.COLOR_BGR2RGB)
img_s=Image.fromarray(img_for_masking, "RGB")
img_s.save("cat_for_masking.jpg")

img_after_masking=cv2.imread("C:/Users/jothi/Downloads/cat_after_masking.jpg")
mask_dif=cv2.subtract(img_after_masking, img_without_mask)

mask_grey=cv2.cvtColor(mask_dif, cv2.COLOR_BGR2GRAY)

ret, threshold_img=cv2.threshold(mask_grey, 30, 255, 0) #0 here refers to the mode code

mask[threshold_img==255]=0 #adjusting the mask values
mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 15, cv2.GC_INIT_WITH_MASK)
mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

img_with_mask = img*mask2[:,:,np.newaxis] #broadcasting

plt.subplot(222)
plt.imshow(cv2.cvtColor(img_without_mask, cv2.COLOR_BGR2RGB))
plt.title("grabcut-without mask")
plt.xticks([])
plt.yticks([])

plt.subplot(221)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("original")
plt.xticks([])
plt.yticks([])

plt.subplot(223)
plt.imshow(cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2RGB))
plt.title("masking-region")
plt.xticks([])
plt.yticks([])

plt.subplot(224)
plt.imshow(cv2.cvtColor(img_with_mask, cv2.COLOR_BGR2RGB))
plt.title("grabcut-with-mask")
plt.xticks([])
plt.yticks([])

plt.show()