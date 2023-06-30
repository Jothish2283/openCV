import cv2

erode_k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
dialate_k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
# print(erode_k, "\n\n", dialate_k)

vid_c=cv2.VideoCapture(0)
for i in range(45):
    s,f=vid_c.read() #to ensure that the camera has auto adjusted the exposure and lighting
    
bg=cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
bg_blurred=cv2.GaussianBlur(bg, (21,21), 0)

s,f=vid_c.read()
while s:
    scene=cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    scene_blurred=cv2.GaussianBlur(scene, (21,21), 0)
    
    diff=cv2.absdiff(bg_blurred, scene_blurred)
    _, thresh=cv2.threshold(diff, 40, 255, 0)
    cv2.erode(thresh, erode_k, thresh, iterations=2)
    cv2.dilate(thresh, dialate_k, thresh, iterations=2)
    
    c,h=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(f, c, -1, (0,255,0), 2)
    for con in c:
        if cv2.contourArea(con)>1500:
            x,y,w,h=cv2.boundingRect(con)
            f=cv2.rectangle(f, (x,y), (x+w,y+h), (0,255,255), 2)
    cv2.imshow("object_tracking", f)
    cv2.waitKey(1)
    if cv2.waitKey(3)==27:
        cv2.destroyAllWindows()
        vid_c.release()
        break
    s,f=vid_c.read()