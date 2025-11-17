import streamlit as st
from PIL import Image
import cv2
import numpy as np
from pipeline import load_models, run_inference_pipeline
from price_map import CAKE_PRICES_MAP, format_currency

st.set_page_config(page_title="Auto Bakery", layout="wide")

st.markdown("""
<style>
[data-testid="stApp"] {
    background-color: #8a7f70; 
    color: #333333; 
    font-size: 1.1rem; 
}

[data-testid="stSidebar"] {
    background-color: #a79c8e;
    padding: 1rem 1.5rem; 
}
[data-testid="stSidebar"] * {
     color: #333333 !important; 
}
[data-testid="stSidebar"] h1 {
     color: #333333 !important;
     font-weight: 700 !important; 
     text-align: center; 
     margin-bottom: 1rem;
     font-size: 2.0rem !important; 
}

[data-testid="stSidebar"] [data-testid="stRadio"] {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 90%; 
    margin: 0 auto; 
}
[data-testid="stSidebar"] [data-testid="stFileUploader"],
[data-testid="stSidebar"] [data-testid="stCameraInput"] {
    display: flex;
    flex-direction: column;
    align-items: center; 
    width: 90%; 
    margin: 0 auto; 
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] > label,
[data-testid="stSidebar"] [data-testid="stCameraInput"] > label {
    width: 100%;
    text-align: center;
    font-size: 1.1rem; 
}


[data-testid="stApp"] h3 {
    color: #333333; 
    font-weight: 700; 
    font-size: 1.75rem; 
    padding-bottom: 10px;
    margin-top: 0px; 
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: rgba(255,255,255,0.05); 
    border-radius: 10px; 
    padding: 1.5rem 1.5rem 1rem 1.5rem; 
    margin-bottom: 1.5rem;
    border: 1px solid #333333; 
}


[data-testid="stHorizontalBlock"] {
    background-color: #333333; 
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    color: #a79c8e; 
font-weight: 500; 
    box-shadow: 0 2px 5px rgba(0,0,0,0.2); 
    transition: transform 0.2s ease, box-shadow 0.2s ease; 
}
[data-testid="stHorizontalBlock"]:hover {
    transform: scale(1.015); 
    box-shadow: 0 4px 15px rgba(0,0,0,0.25); 
}

[data-testid="stHorizontalBlock"] [data-testid="stMarkdownContainer"] p {
    color: #a79c8e; 
    font-weight: 500;
}


[data-testid="stHorizontalBlock"] .stButton > button {
    background-color: #a79c8e; 
    color: #333333;
    border: none;
    border-radius: 8px;
}
[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background-color: #c3b9ab; 
    color: #000000; 
}

.stButton > button {
    background-color: #333333;
    color: #a79c8e; 
    border: none;
    border-radius: 10px;
    padding: 12px 20px; 
    font-size: 1.1rem; 
    font-weight: 600; 
    transition: background-color 0.3s ease, transform 0.2s ease; 
}
.stButton > button:hover {
    background-color: #555555; 
    color: #c3b9ab; 
    transform: scale(1.03); 
}
.stButton > button:active {
    background-color: #111111;
    color: #a79c8e;
    transform: scale(0.98); 
}

.total-price {
    font-size: 1.5rem !important; 
    font-weight: 700 !important; 
    color: #333333 !important; 
    margin-top: 1rem;
    text-align: right;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #a79c8e; 
    color: #333333; 
    border-radius: 8px;
    border: 2px solid #333333; 
    font-weight: 600; 
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
    border-color: #000000; 
    box-shadow: 0 2px 10px rgba(0,0,0,0.15); 
}
[data-testid="stSelectbox"] svg {
    fill: #333333; 
}

div[data-baseweb="popover"] ul {
    background-color: #a79c8e; 
    border: 1px solid #333333; 
}
div[data-baseweb="popover"] ul li {
    color: #333333; 
    font-weight: 500; 
}
div[data-baseweb="popover"] ul li:hover {
    background-color: #6a604f; 
    color: #EAE0D5; 
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] button,
[data-testid="stSidebar"] [data-testid="stCameraInput"] button {
background-color: #6a604f; 
    color: #EAE0D5; 
    border-radius: 10px;
    border: none;
    font-weight: 600; 
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover,
[data-testid="stSidebar"] [data-testid="stCameraInput"] button:hover {
    background-color: #50483d; 
     color: #ffffff;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
[data-testid="stSidebar"] [data-testid="stCameraInput"] label {
    color: #333333 !important; 
    font-weight: 600 !important; 
    font-size: 1.1rem; 
}
</style>
""", unsafe_allow_html=True)

if "cart" not in st.session_state:
    st.session_state.cart = []

if "current_image" not in st.session_state:
    st.session_state.current_image = None
    
if "run_inference" not in st.session_state:
    st.session_state.run_inference = False


@st.cache_resource
def load_all_models():
    return load_models()

detector, unet, classifier = load_all_models()

def on_process_click():
    if st.session_state.current_image is not None:
        st.session_state.run_inference = True
        st.session_state.cart = [] 
def on_file_change(key):
    st.session_state.run_inference = False
    if st.session_state[key] is not None:
        st.session_state.current_image = Image.open(st.session_state[key]).convert("RGB")
    else:
        st.session_state.current_image = None

st.sidebar.title("🍰 Auto Bakery")
mode = st.sidebar.radio("Chọn chế độ", ["Upload ảnh", "Camera"])

if mode == "Upload ảnh":
    st.sidebar.file_uploader(
        "Upload ảnh bánh", 
        type=["jpg","png","jpeg"],
        key="uploaded_file_key",
        on_change=lambda: on_file_change("uploaded_file_key")
    )
elif mode == "Camera":
    st.sidebar.camera_input(
        "Chụp bánh trực tiếp",
        key="camera_file_key",
        on_change=lambda: on_file_change("camera_file_key")
    )

st.sidebar.button(
    "💰 Tính tiền ngay!", 
    on_click=on_process_click,
use_container_width=True,
    disabled=(st.session_state.current_image is None)
)
st.sidebar.markdown("---") 

if st.session_state.run_inference and st.session_state.current_image is not None:
    out_img, total_price, detected_items = run_inference_pipeline(
        detector, 
        unet,
        classifier, 
        st.session_state.current_image
    )
    st.session_state.cart.extend(detected_items)
    
    st.session_state.run_inference = False 

    out_img_rgb = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    st.image(out_img_rgb, caption="Kết quả tính tiền", use_container_width=True)

else:
    if st.session_state.current_image is not None:
         st.image(np.array(st.session_state.current_image), caption="Sẵn sàng tính tiền...", use_container_width=True)
    
with st.container(border=True):
    st.subheader("🛒 Giỏ hàng")

    def remove_item(index):
        if 0 <= index < len(st.session_state.cart):
            st.session_state.cart.pop(index)

    if st.session_state.cart:
        for i, item in enumerate(st.session_state.cart):
            col1, col2 = st.columns([5,1])
            col1.write(f"{item['name']} - {format_currency(item['price'])}")
            col2.button("❌", key=f"del_{i}", on_click=remove_item, args=(i,))

        total = sum(item['price'] for item in st.session_state.cart)
        st.markdown(f'<p class="total-price"><strong>Tổng tiền:</strong> {format_currency(total)}</p>', unsafe_allow_html=True)
    
    else:
        st.write("Giỏ hàng trống.")
    
    st.write("") 
    
    def reset_cart():
        st.session_state.cart = []
    st.button("🗑️ Xóa toàn bộ giỏ hàng", on_click=reset_cart, use_container_width=True)


with st.container(border=True):
    st.subheader("➕ Thêm bánh")
    new_name = st.selectbox("Chọn bánh", list(CAKE_PRICES_MAP.keys()), key="manual_select_box", label_visibility="collapsed")
    
    def add_manual():
        st.session_state.cart.append({"name": new_name, "price": CAKE_PRICES_MAP[new_name]})
    st.button("Thêm vào giỏ", key="add_manual", on_click=add_manual, use_container_width=True)
