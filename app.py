import streamlit as st
import time
import random

# 1. Cấu hình trang & Giao diện Cyberpunk
st.set_page_config(page_title="Hệ Thống Quét Nhân Phẩm Độc Lập", page_icon="🕵️‍♂️", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #00ff41; }
    .stButton>button {
        width: 100%; border-radius: 5px; border: 1px solid #00ff41;
        background-color: #1a1c23; color: #00ff41;
        font-family: 'Courier New', Courier, monospace;
    }
    .stButton>button:hover { background-color: #00ff41; color: #000; box-shadow: 0 0 15px #00ff41; }
    .glitch {
        font-size: 2rem; font-weight: bold; text-transform: uppercase;
        text-shadow: 0.05em 0 0 rgba(255,0,0,.75), -0.025em -0.05em 0 rgba(0,255,0,.75);
        animation: glitch 500ms infinite;
    }
    @keyframes glitch {
        0% { transform: translate(0); }
        20% { transform: translate(-2px, 2px); }
        40% { transform: translate(-2px, -2px); }
        60% { transform: translate(2px, 2px); }
        80% { transform: translate(2px, -2px); }
        100% { transform: translate(0); }
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Khởi tạo Session State
if 'step' not in st.session_state: st.session_state.step = "identify"
if 'user' not in st.session_state: st.session_state.user = None
if 'q_step' not in st.session_state: st.session_state.q_step = 1

st.markdown('<p class="glitch">CORE SYSTEM: INDIVIDUAL ANALYSIS</p>', unsafe_allow_html=True)

# --- BƯỚC 1: XÁC ĐỊNH ĐỐI TƯỢNG ---
if st.session_state.step == "identify":
    st.write("### ⚠️ TRUY CẬP CÓ ĐIỀU KIỆN")
    st.write("Hệ thống yêu cầu xác định đối tượng để tải cơ sở dữ liệu nhân phẩm riêng biệt.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Truy cập hồ sơ KIỆT"):
            st.session_state.user = "Kiệt"
            st.session_state.step = "questions"
            st.rerun()
    with col2:
        if st.button("Truy cập hồ sơ CHÍNH"):
            st.session_state.user = "Chính"
            st.session_state.step = "questions"
            st.rerun()

# --- BƯỚC 2: CÂU HỎI TRẮC NGHIỆM TÂM LÝ ---
elif st.session_state.step == "questions":
    st.write(f"### 🛡️ Câu hỏi xác minh cho: {st.session_state.user}")
    
    if st.session_state.q_step == 1:
        st.write("**Câu 1: Bạn thường làm gì khi thấy một chàng trai đẹp mã đi ngang qua?**")
        if st.button("A. Nhìn thẳng vào mắt để khiêu khích"): st.session_state.q_step = 2; st.rerun()
        if st.button("B. Lén lút nhìn từ dưới lên trên"): st.session_state.q_step = 2; st.rerun()
        if st.button("C. Giả vờ bấm điện thoại nhưng bật camera trước"): st.session_state.q_step = 2; st.rerun()

    elif st.session_state.q_step == 2:
        st.write("**Câu 2: Nếu được chọn một món quà từ 'người bạn thân nhất', bạn thích gì?**")
        if st.button("A. Một nụ hôn bất ngờ lên má"): st.session_state.step = "scan"; st.rerun()
        if st.button("B. Một cuốn cẩm nang 'Cách thoát kiếp thẳng'"): st.session_state.step = "scan"; st.rerun()
        if st.button("C. Một đêm mất ngủ cùng nhau"): st.session_state.step = "scan"; st.rerun()

# --- BƯỚC 3: QUÉT DỮ LIỆU ---
elif st.session_state.step == "scan":
    st.write(f"### 🔍 Đang trích xuất dữ liệu bí mật của {st.session_state.user}...")
    bar = st.progress(0)
    for i in range(100):
        time.sleep(0.03)
        bar.progress(i + 1)
    st.session_state.step = "result"
    st.rerun()

# --- BƯỚC 4: KẾT QUẢ RIÊNG BIỆT ---
elif st.session_state.step == "result":
    st.success(f"✅ BÁO CÁO PHÂN TÍCH CHO ĐỐI TƯỢNG: {st.session_state.user.upper()}")
    
    if st.session_state.user == "Kiệt":
        st.error("## ĐỘ GAY: 99%")
        st.code(f"""
        [PHÂN TÍCH CHI TIẾT - KIỆT]
        - Trạng thái: Đã 'cong' hoàn toàn, không thể uốn thẳng.
        - Triệu chứng: Hay nhìn trai với ánh mắt đắm đuối.
        - Đặc điểm: Thích xem SEX nhưng tâm hồn hướng về 'cột điện'.
        - Lời khuyên: Đừng gồng nữa Kiệt ơi, Chính biết hết rồi.
        """, language="bash")
        st.warning("⚠️ Cảnh báo: Hệ thống phát hiện Kiệt đang có ý định 'tấn công' Chính vào tối nay.")

    elif st.session_state.user == "Chính":
        st.error("## ĐỘ GAY: 80%")
        st.code(f"""
        [PHÂN TÍCH CHI TIẾT - CHÍNH]
        - Trạng thái: Gay tiềm ẩn (Bán lộ).
        - Triệu chứng: Thích được Kiệt quan tâm nhưng hay giả vờ ngại ngùng.
        - Đặc điểm: Thích 'núi đôi' nhưng thực tế lại hay đi chơi riêng với Kiệt.
        - Lời khuyên: 80% là con số khá cao, Chính nên chuẩn bị tâm lý làm 'cô dâu'.
        """, language="bash")
        st.warning("⚠️ Cảnh báo: Chính đang lọt vào tầm ngắm của đối tượng Kiệt (99% Gay).")

    if st.button("ĐĂNG XUẤT (XÓA DẤU VẾT)"):
        st.session_state.step = "identify"
        st.session_state.q_step = 1
        st.session_state.user = None
        st.rerun()

st.caption("Ghi chú: Kết quả dựa trên thuật toán AI dự đoán nhân phẩm. Không dành cho người nghiêm túc.")
