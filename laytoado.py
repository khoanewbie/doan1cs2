import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# 1. Hiện hộp thoại để bạn dùng chuột TỰ CHỌN FILE ẢNH
root = tk.Tk()
root.withdraw() # Ẩn cửa sổ nền đen của tkinter
print("Đang mở hộp thoại chọn ảnh, bạn hãy nhìn dưới thanh Taskbar nếu không thấy cửa sổ hiện lên nhé...")

# Mở bảng chọn file
file_path = filedialog.askopenfilename(
    title="Chọn ảnh bản đồ của bạn", 
    filetypes=[("Image files", "*.png *.jpg *.jpeg")]
)

if not file_path:
    print("❌ BẠN CHƯA CHỌN ẢNH NÀO!")
else:
    # 2. Đọc ảnh (Dùng numpy để chống mọi loại lỗi tiếng Việt có dấu)
    img_array = np.fromfile(file_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        print("❌ LỖI: Không thể đọc được file này. Vui lòng chọn một file ảnh bình thường.")
    else:
        print("✅ ĐÃ MỞ ẢNH THÀNH CÔNG!")
        print("👉 HÃY CLICK CHUỘT TRÁI VÀO CÁC ĐIỂM TRÊN ẢNH ĐỂ LẤY TỌA ĐỘ...")
        print("👉 Bấm phím ESC trên bàn phím để thoát khi lấy xong.\n")

        # 3. Hàm xử lý click chuột
        def click_event(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                # In ra màn hình để bạn copy
                print(f"({x}, {y})")
                # Vẽ chấm đỏ
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow('Tool Lay Toa Do', img)

        # 4. Hiện ảnh lên
        cv2.imshow('Tool Lay Toa Do', img)
        cv2.setMouseCallback('Tool Lay Toa Do', click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 