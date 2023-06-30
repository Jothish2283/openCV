import cv2

vid_c= cv2.VideoCapture(0)
for i in range(30):
    s,f=vid_c.read() #allowing camer to adjust exposure automatically
    
s,f=vid_c.read()

f_h, f_w=f.shape[:2] #frame height and width
tw_w=f_w//8 #tracking frame width
tw_h=f_h//8
tw_x=f_w//2-tw_w//2
tw_y=f_h//2-tw_h//2
track_w=(tw_x,tw_y,tw_w,tw_h)

roi=f[tw_y:tw_y+tw_h, tw_x:tw_x+tw_w]
roi_hsv=cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

mask=None
roi_hist= cv2.calcHist([roi_hsv], [0], mask, [180], [0,180])

# cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)


term_crit=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10,1)

s,f=vid_c.read()
while s:
    f_hsv=cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    back_proj=cv2.calcBackProject([f_hsv], [0], roi_hist, [0,180], 1)
    num_iters, track_w= cv2.meanShift(back_proj, track_w, term_crit)
    
    x,y,w,h=track_w
    cv2.rectangle(f, (x,y), (x+w,y+h), [0,255,255], 2)
    cv2.imshow("tracking_window", f)
    cv2.waitKey(1)
    cv2.imshow("back_project", back_proj)
    cv2.waitKey(1)
    
    if cv2.waitKey(1)==27:
        vid_c.release()
        cv2.destroyAllWindows()
        break
    s,f=vid_c.read()