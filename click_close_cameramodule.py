import cv2

click=True

def click_close(event, x, y, flags, param):
    global click
    if event== cv2.EVENT_LBUTTONUP:
        click=False
        
vid_c=cv2.VideoCapture(0)
cv2.namedWindow("clickclose")
cv2.setMouseCallback("clickclose", click_close)
print("To close the window press left mouse key")

success, frame=vid_c.read()
while (success and cv2.waitKey(1)!=27 and click): #ASCII 27= ESC button
    cv2.imshow("clickclose", frame)
    success, frame= vid_c.read()
    
cv2.destroyWindow("clickclose")
vid_c.release()