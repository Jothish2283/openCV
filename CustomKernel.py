import numpy as np
import scipy
import cv2

#ker_3c, ker_3, ker_5 are ex of HPF
ker_3co=np.array([[[0,-0.25,0],
                 [-0.25,2,-0.25],
                 [0,-0.25, 0]],
                [[0,-0.25,0],
                 [-0.25,2,-0.25],
                 [0,-0.25, 0]],
                [[0,-0.25,0],
                 [-0.25,2,-0.25],
                 [0,-0.25, 0]]])

ker_3c=np.array([[-1,-1,-1], #contrast boost->sharpening [Σai=1 +ve & -ve enteries]
                [-1,9,-1],
                [-1,-1,-1]])

ker_3e=np.array([[-1,-1,-1], #edge detection [Σai=0]
                [-1,8,-1],
                [-1,-1,-1]])

ker_3blur=np.array([[0.11, 0.11, 0.11], #avg blurring [Σai=1 only +ve enteries]
                    [0.11, 0.11, 0.11],
                    [0.11, 0.11, 0.11]])

ker_3embosse=np.array([[-2, -1, 0], #half blur +ve weights+ half sharpen -ve weights= embosse [Σai=1]
                       [-1,  1, 1],
                       [ 0,  1, 2]])

ker_5=np.array([[1,1,1,1,1],
                [1,2,2,2,1],
                [1,2,5,2,1],
                [1,2,2,2,1],
                [1,1,1,1,1]])

# img=cv2.imread("C:/Users/jothi/Pictures/Saved Pictures/Wallheaven/wallhaven-odq7o5.png", 33)
# img_g=cv2.imread("C:/Users/jothi/Pictures/Saved Pictures/Wallheaven/wallhaven-odq7o5.png", 32)

img=cv2.imread("C:/Users/jothi/Downloads/WhatsApp Image 2023-05-30 at 21.51.53.jpeg", 33)
img_g=cv2.imread("C:/Users/jothi/Downloads/WhatsApp Image 2023-05-30 at 21.51.53.jpeg", 32)

cv2.imshow("img", img)
cv2.waitKey(3000)

k3co=scipy.ndimage.convolve(img, ker_3co)
cv2.imshow("k3color",k3co)
cv2.waitKey(3000)

cv2.imshow("img_g", img_g)
cv2.waitKey(3000)

k3c=scipy.ndimage.convolve(img_g, ker_3c)
cv2.imshow("k3contrast",k3c)
cv2.waitKey(3000)

k3e=scipy.ndimage.convolve(img_g, ker_3e)
cv2.imshow("k3edge",k3e)
cv2.waitKey(3000)

k3blur=scipy.ndimage.convolve(img_g, ker_3blur)
cv2.imshow("k3blur",k3blur)
cv2.waitKey(3000)

k3embosse=scipy.ndimage.convolve(img_g, ker_3embosse)
cv2.imshow("k3embosse",k3embosse)
cv2.waitKey(3000)

k5=scipy.ndimage.convolve(img_g, ker_5)
cv2.imshow("k5",k5)
cv2.waitKey(3000)

g_blur=cv2.GaussianBlur(img_g, (17,17), 0) #blurring uses LPF, (17,17)->kernel size, 0->std deviation
cv2.imshow("Gaussian_Blur_LPF", g_blur)
cv2.waitKey(3000)

g_blur_hpf=img_g-g_blur
cv2.imshow("Gaussian_Blur_HPF", g_blur_hpf)
cv2.waitKey(8000)

cv2.destroyAllWindows() 