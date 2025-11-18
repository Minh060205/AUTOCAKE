import streamlit as st
from PIL import Image
import cv2
from pipeline import load_models, run_inference_pipeline
from price_map import CAKE_PRICES_MAP, format_currency

st.set_page_config(page_title="Auto Bakery", layout="wide")

st.markdown("""
<style>
[data-testid="stApp"] { background-color: #8a7f70; color: #333333; font-size: 1.1rem; }
[data-testid="stSidebar"] { background-color: #a79c8e; padding: 1rem 1.5rem; }
[data-testid="stSidebar"] * { color: #333333 !important; }
[data-testid="stSidebar"] h1 { color: #333333 !important; font-weight: 700 !important; text-align: center; margin-bottom: 1rem; font-size: 2.0rem !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] { display: flex; flex-direction: column; align-items: center; width: 90%; margin: 0 auto; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { white-space: nowrap; }
[data-testid="stSidebar"] [data-testid="stFileUploader"],
[data-testid="stSidebar"] [data-testid="stCameraInput"] { display: flex; flex-direction: column; align-items: center; width: 90%; margin: 0 auto; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] > label,
[data-testid="stSidebar"] [data-testid="stCameraInput"] > label { width: 100%; text-align: center; font-size: 1.1rem; }
[data-testid="stSidebar"] .stButton > button { background-color: #6a604f; color: #EAE0D5; font-weight: 700; }
[data-testid="stSidebar"] .stButton > button:hover { background-color: #50483d; color: #ffffff; }
[data-testid="stApp"] > div:not([data-testid="stSidebar"]) .stButton > button[kind="primary"] { background-color: #6a604f !important; color: #EAE0D5 !important; border: none !important; font-weight: 700; }
[data-testid="stApp"] > div:not([data-testid="stSidebar"]) .stButton > button[kind="primary"]:hover { background-color: #50483d !important; color: #ffffff !important; }
[data-testid="stApp"] h3 { color: #333333; font-weight: 700; font-size: 1.75rem; padding-bottom: 10px; margin-top: 0px; }
[data-testid="stVerticalBlockBorderWrapper"] { background-color: rgba(255,255,255,0.05); border-radius: 10px; padding: 1.5rem 1.5rem 1rem 1.5rem; margin-bottom: 1.5rem; border: 1px solid #333333; }
[data-testid="stHorizontalBlock"] { background-color: #333333; padding: 12px; border-radius: 10px; margin-bottom: 10px; color: #a79c8e; font-weight: 500; box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: transform 0.2s ease, box-shadow 0.2s ease; }
[data-testid="stHorizontalBlock"]:hover { transform: scale(1.015); box-shadow: 0 4px 15px rgba(0,0,0,0.25); }
[data-testid="stHorizontalBlock"] [data-testid="stMarkdownContainer"] p { color: #a79c8e; font-weight: 500; }
[data-testid="stHorizontalBlock"] .stButton > button { background-color: #a79c8e; color: #333333; border: none; border-radius: 8px; }
[data-testid="stHorizontalBlock"] .stButton > button:hover { background-color: #c3b9ab; color: #000000; }
[data-testid="stVerticalBlockBorderWrapper"] .stButton { display: flex; justify-content: flex-end; }
[data-testid="stVerticalBlockBorderWrapper"] .stButton > button { background-color: #a77f5e !important; color: #FFFFFF !important; border: none !important; width: auto !important; flex-grow: 0 !important; border-radius: 10px; padding: 12px 20px; font-size: 1.1rem; font-weight: 600; transition: background-color 0.3s ease, transform 0.2s ease; }
[data-testid="stVerticalBlockBorderWrapper"] .stButton > button:hover { background-color: #6a604f !important; color: #EAE0D5 !important; border: none !important; transform: scale(1.03); }
[data-testid="stVerticalBlockBorderWrapper"] .stButton > button:active { background-color: #50483d !important; color: #ffffff !important; transform: scale(0.98); }
[data-testid="stVerticalBlockBorderWrapper"] .stButton > button[kind="primary"] { background-color: #a77f5e !important; color: #FFFFFF !important; border: none !important; }
[data-testid="stVerticalBlockBorderWrapper"] .stButton > button[kind="primary"]:hover { background-color: #6a604f !important; color: #EAE0D5 !important; border: none !important; }
[data-testid="stVerticalBlockBorderWrapper"] .stButton > button[kind="secondary"] { background-color: #a77f5e !important; color: #FFFFFF !important; border: none !important; }
[data-testid="stVerticalBlockBorderWrapper"] .stButton > button[kind="secondary"]:hover { background-color: #6a604f !important; color: #EAE0D5 !important; border: none !important; }
.total-price { font-size: 1.5rem !important; font-weight: 700 !important; color: #333333 !important; margin-top: 1rem; text-align: right; }
</style>
""", unsafe_allow_html=True)

if "cart" not in st.session_state: st.session_state.cart = []
if "processed_image" not in st.session_state: st.session_state.processed_image = None
if "image_to_process" not in st.session_state: st.session_state.image_to_process = None
if "view" not in st.session_state: st.session_state.view = "cart"
if "payment_done" not in st.session_state: st.session_state.payment_done = False
if "payment_method" not in st.session_state: st.session_state.payment_method = None

@st.cache_resource
def load_all_models():
    return load_models()
detector, unet, classifier = load_all_models()

def go_to_payment(): st.session_state.view = "payment"
def go_to_cart(): st.session_state.view = "cart"

def process_payment(method, total_amount):
    st.balloons()
    st.toast(f"Đã thanh toán {format_currency(total_amount)} bằng {method}!", icon="✅")
    st.session_state.cart = []
    st.session_state.processed_image = None
    st.session_state.image_to_process = None
    st.session_state.view = "cart"

st.sidebar.title("🍰 Auto Bakery")
mode = st.sidebar.radio("Chọn chế độ", ["Upload ảnh", "Camera"])

image_file = None
if mode == "Upload ảnh":
    image_file = st.sidebar.file_uploader("Upload ảnh bánh", type=["jpg","png","jpeg"], label_visibility="collapsed")
elif mode == "Camera":
    image_file = st.sidebar.camera_input("Chụp bánh trực tiếp", label_visibility="collapsed")

if image_file:
    st.session_state.image_to_process = image_file

if st.sidebar.button("💰 Tính tiền! 💰", use_container_width=True, type="primary"):
    if st.session_state.image_to_process:
        pil_image = Image.open(st.session_state.image_to_process).convert("RGB")
        out_img_cv, _, detected_items = run_inference_pipeline(detector, unet, classifier, pil_image)
        st.session_state.cart = detected_items
        st.session_state.processed_image = cv2.cvtColor(out_img_cv, cv2.COLOR_BGR2RGB)
    else:
        st.sidebar.warning("Vui lòng upload hoặc chụp ảnh trước!")

if st.session_state.view == "cart":
    if st.session_state.processed_image is not None:
        st.image(st.session_state.processed_image, caption="Kết quả tính tiền...", use_container_width=True)
    elif st.session_state.image_to_process is not None:
        st.image(st.session_state.image_to_process, caption="Ảnh đang chờ xử lý...", use_container_width=True)
    else:
        st.image("https://placehold.co/1600x900/8a7f70/333333?text=Chao+mung+quy+khach+!", caption="Chưa có ảnh", use_container_width=True)

    with st.container(border=True):
        st.subheader("🛒 Giỏ hàng")
        def remove_item(index):
            if 0 <= index < len(st.session_state.cart): st.session_state.cart.pop(index)
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
        def reset_cart(): st.session_state.cart = []
        st.button("🗑️ Xóa toàn bộ giỏ hàng", on_click=reset_cart)

    with st.container(border=True):
        st.subheader("➕ Thêm bánh")
        new_name = st.selectbox("Chọn bánh", list(CAKE_PRICES_MAP.keys()), label_visibility="collapsed")
        def add_manual(): st.session_state.cart.append({"name": new_name, "price": CAKE_PRICES_MAP[new_name]})
        st.button("Thêm vào giỏ", key="add_manual", on_click=add_manual)

    if st.session_state.cart:
        st.button("✅ Hoàn tất đơn hàng", on_click=go_to_payment, use_container_width=True, type="primary")

elif st.session_state.view == "payment":
    with st.container(border=True):
        st.button("⬅️ Quay lại giỏ hàng", on_click=go_to_cart)
        st.subheader("💳 Thanh toán")
        st.divider()
        total = sum(item['price'] for item in st.session_state.cart)
        st.markdown(f'## Tổng cộng: <span style="font-weight: 700; color: #a77f5e; float: right;">{format_currency(total)}</span>', unsafe_allow_html=True)
        st.write("")
        st.subheader("Chọn phương thức thanh toán:")
        col1, col2, col3 = st.columns(3)

        def pay_cash(): st.session_state.payment_done = True; st.session_state.payment_method="Tiền mặt"
        def pay_card(): st.session_state.payment_done = True; st.session_state.payment_method="Thẻ"
        def pay_transfer(): st.session_state.payment_done = True; st.session_state.payment_method="Chuyển khoản"

        with col1: st.button("💵 Tiền mặt", on_click=pay_cash, use_container_width=True, type="primary")
        with col2: st.button("💳 Thẻ", on_click=pay_card, use_container_width=True, type="primary")
        with col3: st.button("📲 Chuyển khoản", on_click=pay_transfer, use_container_width=True, type="primary")

        if st.session_state.payment_done:
            st.write("")
            st.subheader("📄 Hóa đơn")
            for item in st.session_state.cart:
                st.write(f"- {item['name']} : {format_currency(item['price'])}")
            st.markdown(f'**Tổng cộng:** {format_currency(total)}')

        if st.session_state.payment_method == "Chuyển khoản":
            st.write("Quét QR để thanh toán:")
            qr_img = Image.open(r"C:\Users\ADMIN\Downloads\autocake_aicuoiky\qrcode.jpg")
            st.image(qr_img, width=200)

        if st.button("✅ Xác nhận thanh toán", use_container_width=True, type="primary"):
            process_payment(st.session_state.payment_method, total)
            st.session_state.payment_done = False
            st.session_state.payment_method = None
