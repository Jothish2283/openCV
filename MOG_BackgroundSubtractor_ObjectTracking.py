import cv2

bg_subtractor=cv2.createBackgroundSubtractorMOG2(detectShadows=True)
erode_k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,5)) #(3,5)->(w,h)
dilate_k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,9))

# vid_c=cv2.VideoCapture("C:/Users/jothi/Downloads/hallway.mpg")
vid_c=cv2.VideoCapture("C:/Users/jothi/Downloads/supermarket.mp4")
s,f=vid_c.read()

while s:
    mask=bg_subtractor.apply(f)
    _, thresh=cv2.threshold(mask, 244, 255, 0)
    cv2.erode(thresh, erode_k, thresh, iterations=2)
    cv2.dilate(thresh, dilate_k, thresh, iterations=2)
    C,h=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in C:
        if cv2.contourArea(c)>1000:
            x,y,w,h=cv2.boundingRect(c)
            cv2.rectangle(f, (x,y), (x+w, y+h), (0,255,255), 2)
    cv2.imshow("original_frame", f)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("MOG_Mask", mask)
    cv2.waitKey(1)
    if cv2.waitKey(3)==27:
        cv2.destroyAllWindows()
        break
    s,f=vid_c.read()

cv2.destroyAllWindows()