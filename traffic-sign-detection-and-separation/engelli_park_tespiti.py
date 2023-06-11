import torch
import matplotlib.pyplot as plt
import os
import pandas
import cv2
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='bestv5.pt')
dir_path = r'.\\images\\'
ok_file_format = [".jpg", ".png"]

for path in os.listdir(dir_path):

    if path[-4:] in ok_file_format:
        image = cv2.imread(dir_path + path)  # results = model(dir_path + path) 2 satır yerine
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(image)
        Y, X, d = results.render()[0].shape
        df = results.pandas().xyxy[0]
        df = df[df['class'] == 16]
        # print(df[df['class'] == 17])
        threshold = 0.5
        df = df[df['confidence'] >= threshold]
        df = df.drop(['class', 'name', 'confidence'], axis=1)
        df['center_x'] = (df['xmin'] + df['xmax']) // 2
        df['center_y'] = (df['ymin'] + df['ymax']) // 2

        # print(df.loc[0,'center_x'])
        # print(distance_signs)
        # print(df.head())
        df = df.drop(['center_x', 'center_y'], axis=1)

        for i, (xmin, ymin, xmax, ymax) in enumerate(df.values):
            # xmin, ymin, xmax, ymax = row.astype(int)
            xmin = int(max(0, xmin))
            ymin = int(max(0, ymin))
            xmax = int(min(X, xmax))
            ymax = int(min(Y, ymax))
            crop_image = results.render()[0][ymin:ymax, xmin:xmax]

            height, width, channels = crop_image.shape
            mask = np.zeros((height, width), dtype=np.uint8)
            triangle_cords = np.array([[(width, height // 2), (width // 2, height), (width, height)]])
            cv2.fillPoly(mask, triangle_cords, 255)

            crop_image = cv2.bitwise_and(crop_image, crop_image, mask=mask)

            gray_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            _, thresholded = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
            gray_image = _, thresholded
            blurred_image = cv2.GaussianBlur(thresholded, (15, 15), 0)
            print("blur: "+ str(cv2.countNonZero(blurred_image)))
            if cv2.countNonZero(blurred_image) > 0:
                print(path[:-4] + f"_{i}.png =" + " Engelli Park")

            else:
                print(path[:-4] + f"_{i}.png =" + " Park")

            gray_image = blurred_image

            fig, ax = plt.subplots(figsize=(16, 12))
            ax.imshow(gray_image, cmap='gray')
            plt.show()
            plt.savefig(r".\\results\\" + path[:-4] + f"_{i}.png")

        results.save(save_dir=r".\\results\\" + path[:-4])

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()