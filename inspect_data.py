import numpy as np
import os

filename = "benchmark_data.npz"

if not os.path.exists(filename):
    print(f"Lỗi: Không tìm thấy file '{filename}'")
else:
    # 1. Load file
    data = np.load(filename)
    
    print(f"--- ĐANG SOI FILE: {filename} ---")
    
    # 2. Xem danh sách các biến được lưu bên trong (Keys)
    # File .npz giống như một từ điển (dictionary), nó chứa nhiều mảng với tên khác nhau
    print(f"1. Các mảng dữ liệu có trong file: {data.files}")
    
    # 3. In chi tiết từng mảng
    for key in data.files:
        array_data = data[key]
        print(f"\n--- Mảng: '{key}' ---")
        print(f"   + Shape (Kích thước): {array_data.shape}")
        print(f"   + Data Type (Kiểu):   {array_data.dtype}")
        print(f"   + Min/Max Value:      {array_data.min()} / {array_data.max()}")
        print(f"   + Xem thử 3 giá trị đầu (Flatten): {array_data.flatten()[:10]} ...")

    print("\n--- HOÀN TẤT ---")