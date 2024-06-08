import tkinter as tk
from tkinter import *
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageGrab
import tensorflow as tf

# Tải mô hình đã lưu
model = tf.keras.models.load_model('mnist_cnn_model_improved.keras')

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN")

        # Tạo canvas để vẽ chữ số
        self.canvas = Canvas(root, width=200, height=200, bg='white')
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)
        
        # Các nút để thực hiện chức năng xóa và dự đoán
        self.classify_btn = Button(root, text="Dự đoán", command=self.classify_handwriting, font=("Helvetica", 12))
        self.classify_btn.grid(row=1, column=0, pady=2, padx=2)
        
        self.button_clear = Button(root, text="Làm mới", command=self.clear_all, font=("Helvetica", 12))
        self.button_clear.grid(row=1, column=1, pady=2, padx=2)

        # Khởi tạo các biến
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)

    def clear_all(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        if hasattr(self, 'result_label'):
            self.result_label.destroy()

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
        self.draw.line([x1, y1, x2, y2], fill="black", width=10)

    def preprocess_image(self, image):
        # Chuyển đổi hình ảnh thành 28x28 pixel và chuẩn hóa
        image = image.resize((28, 28), Image.BILINEAR)
        image = ImageOps.invert(image)
        image = image.convert('L')
        image = np.array(image)
        image = image.reshape(1, 28, 28, 1).astype('float32') / 255
        return image

    def classify_handwriting(self):
        # Chuyển đổi canvas thành hình ảnh và dự đoán
        digit_image = self.image.copy()
        processed_image = self.preprocess_image(digit_image)
        prediction = model.predict(processed_image)
        digits = np.argsort(prediction[0])[::-1][:3]
        percentages = prediction[0][digits] * 100
        
        self.display_prediction(digits, percentages)


    def display_prediction(self, digits, percentages):
        # Hiển thị kết quả dự đoán
        prediction_text = ""
        for digit, percentage in zip(digits, percentages):
            percentage_str = "{:.2f}".format(percentage)  # Định dạng chuỗi với 4 chữ số thập phân
            prediction_text += f"Số dự đoán: {digit}\nTỉ lệ: {percentage_str}%\n\n"
        self.result_label = Label(self.root, text=prediction_text, font=("Helvetica", 12))
        self.result_label.grid(row=2, column=0, columnspan=2, pady=2, padx=2)

# Tạo ứng dụng
root = Tk()
app = DigitRecognizerApp(root)
root.mainloop()
