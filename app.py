import cv2
import numpy as np
import tensorflow as tf
import src.configs as cf
import time  # Để sử dụng thời gian chờ

MODEL_PATH = "./cnn_asl_model.keras"

# Định nghĩa các lớp ký hiệu ASL
CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "space", "nothing"
]

# Hàm tải mô hình đã huấn luyện
def load_trained_model(model_path):
    """
    Tải model đã huấn luyện từ file .keras
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

# Hàm nhận diện ngôn ngữ ký hiệu
def recognize():
    # Tải mô hình
    model = load_trained_model(MODEL_PATH)
    cam = cv2.VideoCapture(0)  # Mở camera

    text = ""
    word = ""
    count_same_frame = 0
    last_prediction_time = time.time()
    prediction_probability = 0.0

    while True:
        frame = cam.read()[1]
        if frame is None:
            break


        cv2.rectangle(frame, (0, 0), (cf.CROP_SIZE, cf.CROP_SIZE), (0, 255, 0), 3)

        # Tiền xử lý khung hình trước khi đưa vào mô hình
        cropped_image = frame[0:cf.CROP_SIZE, 0:cf.CROP_SIZE]  # Cắt vùng khuôn mặt hoặc tay từ khung hình
        resized_frame = cv2.resize(cropped_image, (cf.IMAGE_SIZE, cf.IMAGE_SIZE))  # Thay đổi kích thước
        reshaped_frame = (np.array(resized_frame)).reshape(
            (1, cf.IMAGE_SIZE, cf.IMAGE_SIZE, 3))  # Định dạng lại khung hình
        frame_for_model = reshaped_frame / 255.0  # Chuẩn hóa khung hình trước khi đưa vào mô hình

        # Chỉ dự đoán nếu đã đủ 1 giây kể từ lần dự đoán trước
        current_time = time.time()
        if current_time - last_prediction_time >= 1:  # Nếu đã 1 giây
            old_text = text  # Lưu trữ giá trị text cũ
            prediction = np.array(model.predict(frame_for_model))  # Dự đoán từ mô hình
            prediction_probability = prediction[0, prediction.argmax()]  # Lấy xác suất của dự đoán có độ tin cậy cao nhất
            text = CLASSES[prediction.argmax()]  # Lấy tên lớp với xác suất cao nhất

            # Nếu dự đoán là 'space', hiển thị dưới dạng '_'
            if text == 'space':
                text = '_'

            # Nếu ký tự không phải là 'nothing'
            if text != 'nothing':
                if old_text == text:  # Nếu ký tự không thay đổi
                    count_same_frame += 1  # Tăng số lượng khung hình giống nhau
                else:
                    count_same_frame = 0  # Đặt lại nếu ký tự thay đổi

                if count_same_frame > 10:  # Nếu ký tự xuất hiện liên tiếp trong 10 khung hình
                    word = word + text  # Thêm ký tự vào từ
                    count_same_frame = 0  # Đặt lại bộ đếm

            last_prediction_time = current_time  # Cập nhật thời gian dự đoán cuối cùng

        # Tạo bảng đen để hiển thị thông tin dự đoán
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, " ", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0))
        cv2.putText(blackboard, f"Predict: {text}", (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        cv2.putText(blackboard, f"Probability: {prediction_probability * 100:.2f}%", (30, 170),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        cv2.putText(blackboard, word, (30, 300), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))

        # Kết hợp khung hình và bảng đen lại với nhau để hiển thị
        res = np.hstack((frame, blackboard))

        # Hiển thị kết quả
        cv2.imshow("Recognizing gesture", res)

        # Điều khiển bằng phím
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):  # Thoát nếu nhấn phím 'q'
            break
        if k == ord('r'):  # Xóa từ nếu nhấn phím 'r'
            word = ""
        if k == ord("z"):  # Xóa ký tự cuối cùng nếu nhấn phím 'z'
            word = word[:-1]

    cam.release()  # Giải phóng camera
    cv2.destroyAllWindows()  # Đóng tất cả cửa sổ OpenCV


recognize()
