import cv2

vid_path="GokuUI.avi"
vid_c= cv2.VideoCapture(vid_path)

fps=vid_c.get(cv2.CAP_PROP_FPS)
size=(int(vid_c.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vid_c.get(cv2.CAP_PROP_FRAME_WIDTH)))
no_frame=vid_c.get(cv2.CAP_PROP_FRAME_COUNT)
print(fps, size, no_frame)

fourcc=cv2.VideoWriter_fourcc('I','4','2','0')
vid_w=cv2.VideoWriter("Goku.avi", fourcc, fps, size)

grab=vid_c.grab()
i=0
while i<3000:

    retv, frame= vid_c.retrieve() 
    i+=1
    grab=vid_c.grab()

print(retv, frame)
cv2.imshow("3000th frame", frame)
cv2.waitKey(3000)
cv2.destroyAllWindows()

if 5 and frame is not None:
    print(True)
# success, frame=vid_c.read()

# while success:
#     vid_w.write(frame)
#     success, frame=vid_c.read()