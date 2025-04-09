import cv2
from ultralytics import YOLO

model=YOLO("runs/detect/train/weights/best.pt")

cap=cv2.VideoCapture(0)

while True:
    r,frame=cap.read()

    if not r:
        break

    result=model(frame)


    annotated_frame=result[0].plot()
    cv2.imshow("Detection",annotated_frame)
    

    if cv2.waitKey(1) & 0XFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()