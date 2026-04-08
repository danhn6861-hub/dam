import streamlit as st
import pandas as pd
import collections
import numpy as np

# Cấu hình trang - GIỮ NGUYÊN 100%
st.set_page_config(page_title="Bot TX Pro Max", layout="wide")
st.title("🤖 Bot Phân Tích Xác Suất & Biến Cố Pro")

# --- KHỞI TẠO DỮ LIỆU ---
# Sử dụng collections.deque để giới hạn bộ nhớ, tránh tràn RAM/Leak khi chạy 24/7
MAX_DATA = 2000

if 'history' not in st.session_state:
    st.session_state.history = []
if 'bot_preds' not in st.session_state:
    st.session_state.bot_preds = []
if 'outcomes' not in st.session_state:
    st.session_state.outcomes = []
# Khởi tạo tham số Bayesian (Alpha/Beta) để theo dõi phân phối xác suất dài hạn
if 'alpha' not in st.session_state:
    st.session_state.alpha = 1.0
if 'beta' not in st.session_state:
    st.session_state.beta = 1.0

# --- THUẬT TOÁN DỰ ĐOÁN (NÂNG CẤP PRO MAX: MARKOV BẬC CAO + LAPLACE SMOOTHING) ---
def advanced_predict():
    hist = st.session_state.history
    if len(hist) < 10:  # Cần đủ mẫu ban đầu để Bayesian và Markov hoạt động
        return "T"
    
    # 1. Thuật toán Markov bậc cao (k=3) với Laplace Smoothing (+1) để tránh xác suất 0
    k = 3
    recent_hist = hist[-150:] # Cửa sổ trượt 150 ván để bắt kịp trend
    pattern = tuple(recent_hist[-k:])
    
    # Đếm tần suất xuất hiện các mẫu hình k ván
    counts = collections.defaultdict(lambda: {'T': 1, 'X': 1}) # Laplace Smoothing khởi tạo 1
    for i in range(len(recent_hist) - k):
        p = tuple(recent_hist[i:i+k])
        nxt = recent_hist[i+k]
        counts[p][nxt] += 1
            
    # Tính xác suất có điều kiện cho mẫu hiện tại
    prob_t_markov = counts[pattern]['T'] / (counts[pattern]['T'] + counts[pattern]['X'])

    # 2. Thành phần Bayesian: Dự đoán dựa trên kỳ vọng hậu nghiệm dài hạn
    prob_t_bayesian = st.session_state.alpha / (st.session_state.alpha + st.session_state.beta)

    # 3. Kết hợp trọng số (Ensemble): Ưu tiên Markov cho ngắn hạn, Bayesian cho dài hạn
    # Trọng số Markov tăng dần theo độ dài lịch sử
    weight_markov = min(0.7, len(hist) / 200)
    final_prob_t = (weight_markov * prob_t_markov) + ((1 - weight_markov) * prob_t_bayesian)

    # Ngưỡng quyết định cực kỳ nhạy
    if final_prob_t > 0.51: return "T"
    if final_prob_t < 0.49: return "X"
    
    # Nếu hòa vốn (50/50), đánh theo xu hướng ván cuối cùng (Trend Following)
    return hist[-1]

# --- XỬ LÝ DỮ LIỆU KHI CLICK (TỐI ƯU CLICK ĂN NGAY) ---
def add_record(res):
    # Lấy dự đoán hiện tại trước khi cập nhật
    current_pred = st.session_state.next_suggestion
    
    # Kiểm tra thắng thua
    status = "✅ THẮNG" if res == current_pred else "❌ THUA"
    
    # Cập nhật tham số Bayesian ngay khi có kết quả thực tế
    if res == 'T':
        st.session_state.alpha += 1
    else:
        st.session_state.beta += 1
    
    # Lưu vào lịch sử và kiểm soát RAM
    st.session_state.history.append(res)
    st.session_state.bot_preds.append(current_pred)
    st.session_state.outcomes.append(status)
    
    if len(st.session_state.history) > MAX_DATA:
        st.session_state.history.pop(0)
        st.session_state.bot_preds.pop(0)
        st.session_state.outcomes.pop(0)

# --- GIAO DIỆN CHÍNH - GIỮ NGUYÊN 100% ---
# Tính toán dự đoán cho ván tiếp theo
st.session_state.next_suggestion = advanced_predict()

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.subheader(f"Dự đoán ván tới: :red[{st.session_state.next_suggestion}]")
    st.write("---")
    st.write("### Nhập kết quả thực tế (Click là ăn ngay):")
    
    # Nút bấm trực tiếp - GIỮ NGUYÊN 100%
    btn_t, btn_x = st.columns(2)
    with btn_t:
        if st.button("🔴 TÀI (T)", key="btn_t", use_container_width=True):
            add_record("T")
            st.rerun()
    with btn_x:
        if st.button("🔵 XỈU (X)", key="btn_x", use_container_width=True):
            add_record("X")
            st.rerun()

# --- THỐNG KÊ BIẾN CỐ - GIỮ NGUYÊN 100% ---
if st.session_state.history:
    total_games = len(st.session_state.history)
    wins = st.session_state.outcomes.count("✅ THẮNG")
    win_rate = (wins / total_games) * 100

    st.sidebar.header("📊 Chỉ số hiệu quả")
    st.sidebar.metric("Tỷ lệ thắng Bot", f"{round(win_rate, 1)}%")
    st.sidebar.write(f"Tổng ván (Lưu trữ): {total_games}")
    
    # Biểu đồ tần suất
    st.sidebar.write("---")
    t_count = st.session_state.history.count("T")
    t_rate = t_count / total_games
    st.sidebar.write(f"Tài (T): {round(t_rate*100)}%")
    st.sidebar.progress(t_rate)
    st.sidebar.write(f"Xỉu (X): {round((1-t_rate)*100)}%")
    st.sidebar.progress(1-t_rate)

# --- HIỂN THỊ LỊCH SỬ DỰ ĐOÁN - GIỮ NGUYÊN 100% ---
st.divider()
st.subheader("📝 Lịch sử dự đoán và Biến cố")

if st.session_state.history:
    display_limit = 50
    df = pd.DataFrame({
        "Ván thứ": range(len(st.session_state.history) - len(st.session_state.history[-display_limit:]) + 1, len(st.session_state.history) + 1),
        "Bot Dự Đoán": st.session_state.bot_preds[-display_limit:],
        "Kết Quả Thực": st.session_state.history[-display_limit:],
        "Trạng Thái": st.session_state.outcomes[-display_limit:]
    })
    
    def color_status(val):
        color = '#2ecc71' if 'THẮNG' in val else '#e74c3c'
        return f'color: {color}; font-weight: bold'

    st.table(df.iloc[::-1].style.applymap(color_status, subset=['Trạng Thái']))

if st.button("🔄 Reset toàn bộ"):
    st.session_state.clear()
    st.rerun()
