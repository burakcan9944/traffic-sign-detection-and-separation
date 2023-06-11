import time
import torch
import matplotlib.pyplot as plt
import cv2
import requests
import numpy as np




model = torch.hub.load('ultralytics/yolov5', 'custom', path='bestV5S_5.pt')
# dir_path = r'.\\images\\'
# ok_file_format = [".jpg", ".png"]qq

cap = cv2.VideoCapture("")
file_str= time.strftime("%Y-%m-%d_%H-%M", time.gmtime())

def color(classs, crop_images):
    if classs==17:

        hsv = cv2.cvtColor(crop_images, cv2.COLOR_RGB2HSV)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        lower_red = np.array([160, 100, 100])
        upper_red = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        mask_red = mask1 + mask2
        red = cv2.countNonZero(mask_red)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow = cv2.countNonZero(mask_yellow)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask3 = cv2.inRange(hsv, lower_green, upper_green)
        lower_green = np.array([25, 50, 50])
        upper_green = np.array([70, 255, 255])
        mask4 = cv2.inRange(hsv, lower_green, upper_green)
        mask_green = mask3 + mask4
        green = cv2.countNonZero(mask_green)
        # time.sleep(2)
        print(red, yellow, green)
        if red > yellow and red > green:
            print("Kırmızı")
        elif yellow > red and yellow > green:
            print("Sarı")
        elif green > red and green > yellow:
            print("Yeşil")
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(crop_images)
        plt.show()
        #cv2.imshow("hsv",hsv)
    else:
        pass

def go_right_left(classs,crop_image):
    if classs == 7 or classs == 8:
        height, width, channels = crop_image.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        center_point = (width * 7 // 10, height * 4 // 10)
        triangle_cords = np.array([[center_point, (width * 2 // 3, height * 1 // 4), (width , height // 2)]])
        cv2.fillPoly(mask, triangle_cords, 255)

        crop_image_masked = cv2.bitwise_and(crop_image, crop_image, mask=mask)
        gray_image = cv2.cvtColor(crop_image_masked, cv2.COLOR_BGR2GRAY)

        _, thresholded = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(thresholded)

        print("White pixels:", white_pixels)

        if white_pixels > 100:
            print("ileri saga mecburi yon")
        else:
            print("ileri sola mecburi yon")

        # Plot the masked image and the triangle region
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))
        ax1.imshow(cv2.cvtColor(crop_image_masked, cv2.COLOR_BGR2RGB))
        ax2.imshow(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
        cv2.imshow("crop_image_masked", crop_image_masked)
        cv2.imshow("image", crop_image)
        plt.show()

def mecburiler(classs,crop_image):
    if classs==9 or classs==10:

        height, width, channels = crop_image.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        #center_point = (width //2, height //2)
        triangle_cords = np.array([[(width//2, height*2//10), (width *2//10 , height*4//10), (width //2, height *4//10)]]) #ortadaki üst nokta,en sağdaki en sağ nokta
        cv2.fillPoly(mask, triangle_cords, 255)
        crop_image = cv2.bitwise_and(crop_image, crop_image, mask=mask)

        gray_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
        #gray_image = _, thresholded
        blurred_image = cv2.GaussianBlur(thresholded, (15, 15), 0)
        cv2.imshow("blur", blurred_image)

        

        print("blur: "+ str(cv2.countNonZero(blurred_image)))
        if cv2.countNonZero(blurred_image) > 150:
            print("sola mecburi")
        else:
            print("sağa mecburi")

        gray_image = blurred_image

        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(gray_image, cmap='gray')



def ileri_ve_veliler(classs,crop_image):
    if classs==5 or classs==6:

        height, width, channels = crop_image.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        #center_point = (width //2, height //2)
        triangle_cords = np.array([[(width* 6//10, height* 4//10), (width //2 ,height*2//10 ), (width *8//10, height //2)]]) #ortadaki üst nokta,en sağdaki en sağ nokta
        cv2.fillPoly(mask, triangle_cords, 255)
        crop_image = cv2.bitwise_and(crop_image, crop_image, mask=mask)

        gray_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
        blurred_image = cv2.GaussianBlur(thresholded, (15, 15), 0)
        cv2.imshow("blur", blurred_image)

        

        print("blur: "+ str(cv2.countNonZero(blurred_image)))
        if cv2.countNonZero(blurred_image) > 200:
            print("ileri ve sola mecburi")
        else:
            print("ileri ve sağa mecburi")

        gray_image = blurred_image

        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(gray_image, cmap='gray')




def no_turn(classs,crop_image):
    if classs == 13 or classs == 14:
        height, width, channels = crop_image.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        center_point = (width *35//100, height *3//10)
        triangle_cords = np.array([[center_point, (width * 1 // 4, height*2//10), (width *3//10, height* 3// 10)]]) #ortadaki üst nokta,en sağdaki en sağ nokta
        cv2.fillPoly(mask, triangle_cords, 255)
        crop_image = cv2.bitwise_and(crop_image, crop_image, mask=mask)

        gray_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
        blurred_image = cv2.GaussianBlur(thresholded, (15, 15), 0)
        cv2.imshow("blur", blurred_image)


        print("blur: "+ str(cv2.countNonZero(blurred_image)))
        if cv2.countNonZero(blurred_image) > 5:
            print("sola donulmez")
        elif cv2.countNonZero(blurred_image) < 5:
            print("saga donulmez")



        gray_image = blurred_image

        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(gray_image, cmap='gray')


def handicapped_park(classs,crop_image):
    if classs==16:
        height, width, channels = crop_image.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        triangle_cords = np.array([[(width, height // 2), (width // 2, height), (width, height)]])
        cv2.fillPoly(mask, triangle_cords, 255)

        crop_image = cv2.bitwise_and(crop_image, crop_image, mask=mask)

        gray_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
        #gray_image = _, thresholded
        blurred_image = cv2.GaussianBlur(thresholded, (15, 15), 0)
        cv2.imshow("blur", blurred_image)

        print("blur: "+ str(cv2.countNonZero(blurred_image)))
        if cv2.countNonZero(blurred_image) > 300:
            print("Engelli Park")
        else:
            print("Park")

        gray_image = blurred_image

        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(gray_image, cmap='gray')


def wait(seconds):

    #start_time = time.perf_counter() #start time
    start_time = time.time()
    print('start', start_time)
    #end_time = time.perf_counter()  #end time
    end_time = seconds
    #elapsed_time = end_time - start_time
    elapsed_time = start_time + end_time
    print('end', elapsed_time)

    print("Görev başlatıldı. Başlama zamanı:", start_time)
    while time.time() < elapsed_time:
        pass

    # Görev tamamlandı mesajı
    print("Görev tamamlandı. Tamamlanma zamanı:", time.time())

def getPreds(depth_map,xmin,xmax,ymin,ymax,classs):
    center_x = (xmin+xmax)/2
    center_y = (ymin+ymax)/2

    frame_z=depth_map[center_x,center_y]

    preds = [classs, center_x, frame_z]

    preds.append(preds)

def stop_order_detection(classs):
    if classs == 0:
        stop = True
        while stop:
                if stop == True:
                    print("Araç tabelaya yaklaştı!")
                    wait(5)
                    print("Araç 5 saniye boyunca durdu.")
                    stop = False


                else:
                    stop = True
    else:
        pass

def station_stop(classs):
    if classs==15:
        d=True
        while d:
            if d==True:
                print("Araç cebe girdi")
                wait(1)
                print("Araç 40 saniye boyunca durdu.")
                d=False
    else:
        pass

def station_open(classs,image,xmin,ymin,xmax,ymax,height,width,distance):
    if classs==15:

        frame_centr_y = height
        frame_centr_x = width

        #center_y = (ymin + ymax) // 2
        center_x = (xmin + xmax) // 2

        # Kameranın merkez noktasını trafik levhasının merkez noktasına öteleme işlemini hesaplayın.
        #delta_y = center_y - frame_centr_y
        delta_x = center_x - frame_centr_x

        # while saga_direksiyon:
        # Kameranın yeni konumunu hesaplayın.
        new_camera_center = (frame_centr_x + delta_x)
        print("New camera center:", new_camera_center)

        if distance <= 1:
            print("Durak levhasi aracın önünde kaldi.")
            if classs==15:
                print("Araç levhayı goruyor ve sola donmeye devam ediyor")
                #sola_dondur()
            elif classs!=15:
                print("Araç levhayı artık gormuyor")


        if  center_x < frame_centr_x:
            print("Durak levhasi sagda kaldi.")
        elif center_x > frame_centr_x:
            print("Durak levhasi solda kaldi.")
        else:
            print("Durak levhasi merkezde kaldi.")

    else:
        pass




array=np.random.rand(720,1280)*10
while True:
    ret, image = cap.read()
    #print(height, width)
    image= cv2.flip(image, -1)
    if not ret:
        break
    #img_resp = requests.get("http://192.168.178.5:8080//shot.jpg")
    #img_arry = np.array(bytearray(img_resp.content), dtype=np.uint8)
    #image = cv2.imdecode(img_arry, cv2.IMREAD_COLOR)
    image=cv2.resize(image,(640,640))
    images = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(images)
    Y, X, d = results.render()[0].shape
    df = results.pandas().xyxy[0]

    #print(df)
    #df = df[df['class'] == 17]
    # print(df[df['class'] == 17])


    #filename = f"{timestamp}.txt"


        #df_filtered = df[['confidence', 'class','name']]
        #f.write(str(df_filtered) + '\n')




    #df['center_x'] = (df['xmin'] + df['xmax']) // 2
    #df['center_y'] = (df['ymin'] + df['ymax']) // 2
    #distance_matrix = np.zeros((720, 1280))
    #distance_signs = distance_traffic_signs(df, distance_matrix)
    # print(df.loc[0,'center_x'])
    #print(len(distance_signs))
    # print(df.head())
    #df = df.drop(['center_x', 'center_y'], axis=1)



    # cv2.imshow("crop_image", crop_image)

    for i, row in df.iterrows(): #row ile birden fazla değer atanabilirim
        classs= row['class']
        threshold = 0.7
        confidence = row['confidence']
        name= row['name']
        xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)

        if len(df.index) > 0:

            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            if array[center_y, center_x] <= 5 and confidence >= threshold:
                distance = array[center_y, center_x]
                print(center_x, center_y, name, confidence, distance, end=' /// ')
            else:
                continue

        print()
        print()
        print()


        #print(df.values)
        # xmin, ymin, xmax, ymax = row.astype(int)

        xmin = int(max(0, xmin))
        ymin = int(max(0, ymin))
        xmax = int(min(X, xmax))
        ymax = int(min(Y, ymax))
        crop_images = results.render()[0][ymin:ymax, xmin:xmax]

        # print(crop_image)

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())

        with open(f'{file_str}', 'a') as f:
            if len(df) > 0:
                f.write("Tarih:  " + str(timestamp) + '\n')
                f.write("Isim: " + str(name) + '\n')
                f.write("Dogruluk Degeri: " + str(confidence) + '\n')
                f.write("ID: " + str(classs) + '\n' + '\n')

        color(classs, crop_images)
        #station_open(classs,image,xmin,ymin,xmax,ymax,Y,X,distance)
        no_turn(classs,crop_images)
        #stop_order_detection(classs)
        #station_stop(classs)
        handicapped_park(classs,crop_images)
        go_right_left(classs,crop_images)
        mecburiler(classs,crop_images)
        ileri_ve_veliler(classs,crop_images)

        # plt.savefig(r".\\results\\" +  f"_{i}.png")
        #cv2.imshow("crop_image", crop_images)
    cv2.imshow("video", image)

    # results.save(save_dir=r".\\results\\" )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

