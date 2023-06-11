import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='bestV5S_3.pt')

cap = cv2.VideoCapture("C:/Users/bccf/OneDrive/Masaüstü/burak videolar/yeni/VID_20230411_154703.mp4")

while True:
    ret, image = cap.read()
    image=cv2.flip(image,-1)
    image = cv2.resize(image, (640, 640))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image)
    Y, X, d = results.render()[0].shape
    df = results.pandas().xyxy[0]
    threshold = 0.7
    df = df[df['confidence'] >= threshold]
    df= df.drop(['xmin','ymin','xmax','ymax','class','confidence'], axis=1)

    print(df)
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(results.render()[0])
    plt.show()
    cv2.imshow("image", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()