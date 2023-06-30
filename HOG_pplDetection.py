import cv2
import numpy as np

def i_in_o(i,o):
    ix,iy,iw,ih=i
    ox,oy,ow,oh=o
    return ix>ox and iy>oy and ix+iw<ox+ow and iy+ih<oy+oh

hog=cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
img= cv2.imread("C:/Users/jothi/Downloads/fam.jpg", 33)
ppl, confs=hog.detectMultiScale(img, winStride=(1,1), scale=1.2)
true_ppl=ppl.copy()
true_confs=confs.copy()

for idx,p1 in enumerate(ppl):
    for p2 in ppl:
        if p1.all()!=p2.all():
            if i_in_o(p1,p2): 
                true_ppl=np.delete(true_ppl, idx, axis=0)
                true_confs=np.delete(true_confs, idx)

for idx, p in enumerate(true_ppl):
    x,y,w,h=p
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    text=f"{confs[idx] :.2f}"
    cv2.putText(img, text, (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
cv2.imshow("marked_ppl", img)
cv2.waitKey(5000)
cv2.destroyAllWindows()