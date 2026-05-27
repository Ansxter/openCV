import cv2
import os


input_folder = "face"
output_folder = "face_output"


os.makedirs(output_folder, exist_ok=True)


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


for filename in os.listdir(input_folder):


    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):

        img_path = os.path.join(input_folder, filename)


        img = cv2.imread(img_path)


        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )


        for (x, y, w, h) in faces:


            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


            cv2.putText(
                img,
                'face',
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)

        print(f"已處理：{filename}")

print("全部圖片處理完成，結果已輸出至 face_output 資料夾")