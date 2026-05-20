import streamlit as st
import cv2
import networkx as nx
import numpy as np
import math
import os
from ultralytics import YOLO
from PIL import Image
import qrcode
import socket

# ==========================================
# 0. CẤU HÌNH ĐƯỜNG DẪN TỚI FILE CỦA BẠN
# ==========================================
IMAGE_PATH = r"C:\Users\DANG KHOA\OneDrive\Máy tính\map cs2.png"
MODEL_PATH = r"C:\Users\DANG KHOA\Downloads\best (6).pt"

@st.cache_resource
def load_yolo_model():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    return None

model = load_yolo_model()

# ==========================================
# 1. DỮ LIỆU BẢN ĐỒ
# ==========================================
nodes = {
    "TDTT-1": (338, 114),
    "TDTT-1.1": (324, 133),
    "TDTT-2": (446, 180),
    "TDTT": (448, 119),
    
    "H3": (311, 255),
    "H3GOC": (446, 235),
    "H3-1": (259, 186),
    "H3-2": (314, 258),
    "H3-3": (417, 276),
    "H3-4": (364, 186),
    "NGATU1": (387, 332),
    "NGATU2": (422, 354),
    
    "H2-1": (204, 280),
    "H2-2": (262, 353),
    "H2-3": (355, 379),
    "H2-4": (304, 288),
    
    "H1-1": (79, 334),
    "H1-2": (184, 449),
    "H1-3": (308, 454),
    "H1-4": (248, 366),
    "H1GOC": (46, 396),
    "H1-2.1": (167, 476),
    "NGABA1": (256, 532),
    "NGABA1.1": (286, 552),
    "NGABA2": (328, 422),
    "H1H2": (118, 284),
    "H1UNI": (182, 318),
    "H2NHAXE": (245, 247),
    
    "H6-1": (506, 443),
    "H6-2": (385, 491),
    "H6-3": (558, 541),
    "H6-4": (401, 632),
    "TRUOCH6": (360, 453),
    "H6": (414, 487),
    "H612": (403, 388),
    "H613": (542, 504),
    "H634": (484, 593),
}

edges = [
    ("TDTT-1", "TDTT-2"),
    ("TDTT-1", "TDTT-1.1"),
    ("TDTT", "TDTT-2"),
    ("TDTT-1.1", "H3-1"),

    ("H3-1", "H2NHAXE"),
    ("H2NHAXE", "H3-2"),
    ("TDTT-1", "H3-4"),
    ("H3GOC", "TDTT-2"),
    ("H3GOC", "H3-4"),
    ("H3GOC", "H3-3"),
    ("H3-2", "H2-4"),

    ("H2-4", "NGATU1"),
    ("NGATU1", "NGATU2"),
    ("NGATU1","H3-3" ),
    ( "H3-3","H3GOC"),

    ("H1GOC", "H1-1"),
    ("H1-1", "H1H2"),
    ("H1-1", "H1UNI"),
    ("H1GOC", "H1-2.1"),
    ("H1-2.1", "H1-2"),
    ("H1UNI", "H1-4"),
    ("H1-2.1", "NGABA1"),
    ("NGABA1", "NGABA1.1"),
    ("NGABA1", "H1-3"),
    ("H1-3", "NGABA2"),
    ("NGABA2", "H1-4"),
    ("NGABA2","H2-3"),
    ("H1UNI", "H2-1"),
    ("H2NHAXE", "H2-1"),
    ("NGABA2","H2-3"),
    ("H2-3","NGATU1"),
    ("H2-2", "H1-4"),
    ("TRUOCH6", "NGABA2"),
    ("H6-4", "NGABA1.1"),
    ("H6-4", "H6-3"),
    ("H6-3", "H613"),
    ("H6-1", "H613"),
    ("TRUOCH6", "NGABA1.1"),
    ("TRUOCH6", "H612"),   
    ("H612", "NGATU2"),   
    ("H612", "H6-1"),
    ("TRUOCH6", "H6-2"),
    ("H634", "H6-3"),
    ("H634", "H6-4"),
]

YOLO_TO_NODE_MAP = {
    "H1": "H1-2",
    "H2": "H2-2",
    "H3": "H3-2",
    "H6": "H6-2",
    "NHATHIDAU": "TDTT",
    "CANTIN": "H1UNI",
}

# --- TỪ ĐIỂN MỚI: QUY ĐỊNH ĐIỂM ĐỨNG GIỮA 2 TÒA NHÀ KHÁC NHAU ---
DIEM_GIUA_CAC_TOA_NHA = {
    ("H1-1", "H1-2"): "H1GOC",      
    ("H1-1", "H1-4"): "H1H2",  
    ("H2-1", "H2-2"): "H1UNI",
    ("H1-2","H1-3"): "NGABA1",
    ("H2-2","H2-3"): "NGABA2",
    ("H1-2","H1-3"): "NGABA2",
    ("H2-3","H2-4"): "NGATU1",
    ("H3-2","H3-3"): "NGATU1",
    ("H2-3","H2-4"): "NGATU1",
    ("H3-3","H3-4"): "H3GOC",
    ("H3-3","H3-4"): "H3GOC",
    ("H1-1","H1-2"): "H2NHAXE",
    ("H2-1","H2-4"): "H2NHAXE",
    ("H1-1","H1-4"): "TDTT-1.1",
    ("H6-1","H6-2"): "H612",
    ("H6-1","H6-4"): "NGABA1.1",
    ("H6-1","H6-3"): "H613",
    ("H6-3","H6-4"): "H634",
}
# ----------------------------------------------------------------

danh_sach_toa_nha = [
    "H1", "H2", "H3", "H6", "NHATHIDAU", "CANTIN",
]

# ==========================================
# 2. GIAO DIỆN WEB
# ==========================================
st.set_page_config(page_title="Hệ Thống Chỉ Đường Thông Minh", layout="wide")

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

local_ip = get_local_ip()
url = f"http://{local_ip}:8501"

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=8,
    border=2,
)
qr.add_data(url)
qr.make(fit=True)
img_qr = qr.make_image(fill_color="black", back_color="white").get_image()

st.sidebar.markdown("### 📱 Truy cập trên điện thoại")
st.sidebar.image(img_qr, caption=f"Link: {url}")
st.sidebar.info("💡 Lưu ý: Điện thoại và máy tính phải bắt chung mạng Wi-Fi thì mới quét được.")

st.title("🗺️ HỆ THỐNG NHẬN DIỆN VÀ CHỈ ĐƯỜNG")

diem_xuat_phat_ai = None 

# -- PHẦN 1: TẢI ẢNH VÀ NHẬN DIỆN BẰNG YOLO --
st.subheader("📸 1. Chụp/Tải ảnh tòa nhà bạn đang đứng")
uploaded_file = st.file_uploader("Tải ảnh lên để AI nhận diện vị trí...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image_to_predict = Image.open(uploaded_file)
    st.image(image_to_predict, caption="Ảnh bạn vừa tải lên", width=300)
    
    with st.spinner("🤖 AI đang quét để tìm tòa nhà..."):
        results = model.predict(image_to_predict)
        
        detected_list = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[class_id]
                
                if not any(d[0] == class_name for d in detected_list):
                    detected_list.append((class_name, conf))
            
            detected_list.sort(key=lambda x: x[1], reverse=True)

            st.markdown("### 👁️ Danh sách các mặt AI phát hiện:")
            for label, conf in detected_list:
                st.write(f"- Mặt **{label}**: {conf * 100:.1f}%")

            # XỬ LÝ NHIỀU ĐIỂM
            if len(detected_list) >= 2:
                label_1, conf_1 = detected_list[0]
                label_2, conf_2 = detected_list[1]
                
                # Hàm lấy tên gốc của tòa nhà (Cắt bỏ phần -1, -2 nếu có để biết là tòa nào)
                def get_base_name(label):
                    return label.split('-')[0] if "-" in label else label

                toa_1 = get_base_name(label_1)
                toa_2 = get_base_name(label_2)

                # TRƯỜNG HỢP A: Đứng ở GÓC của CÙNG 1 TÒA NHÀ
                if toa_1 == toa_2:
                    delta = conf_1 - conf_2
                    if delta < 0.30 and conf_2 >= 0.15:
                        if "-" in label_1 and "-" in label_2:
                            mat_1 = label_1.split('-')[1]
                            mat_2 = label_2.split('-')[1]
                            cac_mat = sorted([mat_1, mat_2])
                            diem_xuat_phat_ai = f"{toa_1}-{cac_mat[0]}{cac_mat[1]}"
                            st.success(f"📍 Phát hiện bạn ở GÓC XEN GIỮA tòa nhà! Tự động chọn: **{diem_xuat_phat_ai}**")
                        else:
                            diem_xuat_phat_ai = label_1
                            st.success(f"🎯 Nhận diện vị trí: **{diem_xuat_phat_ai}**")
                    else:
                        diem_xuat_phat_ai = label_1
                        st.success(f"🎯 Nhận diện vị trí: **{diem_xuat_phat_ai}**")
                
                # TRƯỜNG HỢP B: Đứng GIỮA 2 TÒA NHÀ KHÁC NHAU
                else:
                    # Gộp thành cặp và sắp xếp lại để tra cứu từ điển không bị ngược thứ tự
                    pair = tuple(sorted([toa_1, toa_2]))
                    giao_diem_dict = {tuple(sorted(k)): v for k, v in DIEM_GIUA_CAC_TOA_NHA.items()}
                    
                    if pair in giao_diem_dict:
                        diem_xuat_phat_ai = giao_diem_dict[pair]
                        st.success(f"📍 Phát hiện bạn đứng GIỮA {toa_1} và {toa_2}! Tự động chốt vị trí: **{diem_xuat_phat_ai}**")
                    else:
                        diem_xuat_phat_ai = label_1
                        st.success(f"🎯 Lấy tòa nhà rõ nhất: **{diem_xuat_phat_ai}**")
                        
            else:
                diem_xuat_phat_ai = detected_list[0][0]
                st.success(f"🎯 Nhận diện vị trí: **{diem_xuat_phat_ai}**")
        else:
            st.error("❌ AI không nhận diện được tòa nhà nào.")

st.markdown("---")

# -- PHẦN 2: CHỈ ĐƯỜNG --
st.subheader("📍 2. Chọn điểm đến và Tìm đường")
col1, col2 = st.columns(2)

with col1:
    if diem_xuat_phat_ai:
        start_node = st.selectbox("📍 Điểm xuất phát (AI tự chốt):", [diem_xuat_phat_ai], disabled=True)
    else:
        start_node = st.selectbox("📍 Chọn điểm xuất phát (Thủ công):", danh_sach_toa_nha)

with col2:
    default_end_index = 1 if len(danh_sach_toa_nha) > 1 else 0
    end_node = st.selectbox("🎯 Chọn điểm đến:", danh_sach_toa_nha, index=default_end_index)

hien_duong_di = st.checkbox("👁️ Hiển thị đường đi màu đỏ", value=True)
hien_chi_tiet = st.checkbox("🔍 Hiển thị chi tiết các nút giao (trạm trung gian)", value=False)

if not os.path.exists(IMAGE_PATH):
    st.error("❌ Không tìm thấy file ảnh bản đồ.")
else:
    img_array = np.fromfile(IMAGE_PATH, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    G = nx.Graph()
    for e in edges:
        p1, p2 = nodes[e[0]], nodes[e[1]]
        dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        G.add_edge(e[0], e[1], weight=dist)

    actual_start_node = YOLO_TO_NODE_MAP.get(start_node, start_node)
    actual_end_node = YOLO_TO_NODE_MAP.get(end_node, end_node)

    if actual_start_node not in nodes:
        st.error(f"⚠️ Thư mục bản đồ chưa có tọa độ pixel cho điểm: '{actual_start_node}'. Hãy bổ sung vào biến `nodes` ở dòng 31.")
    elif actual_end_node not in nodes:
        st.error(f"⚠️ Thư mục bản đồ chưa có tọa độ pixel cho điểm: '{actual_end_node}'. Hãy bổ sung vào biến `nodes` ở dòng 31.")
    else:
        try:
            path = nx.shortest_path(G, source=actual_start_node, target=actual_end_node, weight='weight')
            
            if hien_duong_di:
                for i in range(len(path) - 1):
                    pt1, pt2 = nodes[path[i]], nodes[path[i+1]]
                    cv2.line(img, pt1, pt2, (0, 0, 255), 8) 
                    
                    if hien_chi_tiet:
                        cv2.circle(img, pt1, 10, (255, 0, 0), -1) 
                
                if hien_chi_tiet:
                    st.success(f"✅ Lộ trình chi tiết: {' ➔ '.join(path)}")
                else:
                    st.success(f"✅ Lộ trình: {start_node} ➔ {end_node}")

            cv2.circle(img, nodes[actual_start_node], 15, (255, 0, 0), -1) 
            cv2.circle(img, nodes[actual_end_node], 15, (0, 255, 0), -1)   

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Bản đồ chỉ đường", use_container_width=True)

        except nx.NetworkXNoPath:
            st.error("❌ Không có đường đi nối giữa 2 điểm này!")
        except Exception as e:
            st.error(f"❌ Lỗi xử lý đồ thị: {e}")