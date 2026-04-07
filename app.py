import streamlit as st
import streamlit.components.v1 as components

# 1. Cấu hình trang
st.set_page_config(page_title="Gay Test Ultimate 🌈", layout="centered")

# 2. CSS nền
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1f1c2c 0%, #928dab 100%);
    }
    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 3. HTML/JS/CSS hỗn hợp
game_html = """
<div id="wrapper">
    <div class="header-zone">
        <h2 class="title">AI GENDER SCANNER</h2>
        <p class="subtitle">Premium v2.0.1 - Mobile Optimized</p>
    </div>
    
    <div class="question-zone">
        <h1>Bạn có phải là GAY không? 🌈</h1>
    </div>

    <div id="game-container">
        <button id="yes-btn">CÓ 👍</button>
        <button id="no-btn">KHÔNG ❌</button>
    </div>
</div>

<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;800&display=swap');
    
    #wrapper {
        font-family: 'Inter', sans-serif;
        display: flex;
        flex-direction: column;
        align-items: center;
        color: white;
        padding: 20px;
        text-align: center;
    }

    .title {
        font-size: clamp(24px, 5vw, 32px);
        font-weight: 800;
        background: linear-gradient(to right, #ff00cc, #3333ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .subtitle {
        color: #aaa;
        font-size: 14px;
        margin-bottom: 30px;
    }

    .question-zone h1 {
        font-size: clamp(28px, 8vw, 42px);
        margin-bottom: 40px;
    }

    #game-container {
        position: relative;
        width: 100%;
        max-width: 600px;
        height: 350px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        touch-action: none; /* Ngăn chặn scroll khi chạm trên điện thoại */
    }

    #yes-btn {
        padding: 15px 45px;
        font-size: 20px;
        font-weight: bold;
        color: white;
        background: linear-gradient(45deg, #4CAF50, #8BC34A);
        border: none;
        border-radius: 50px;
        cursor: pointer;
        z-index: 10;
        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
    }

    #no-btn {
        position: absolute;
        padding: 12px 30px;
        font-size: 18px;
        font-weight: bold;
        color: white;
        background: linear-gradient(45deg, #ff4b4b, #ff7676);
        border: none;
        border-radius: 50px;
        cursor: pointer;
        z-index: 20;
        transition: all 0.15s ease-out;
        white-space: nowrap;
    }

    /* Tối ưu cho điện thoại */
    @media (max-width: 480px) {
        #game-container {
            height: 400px;
        }
        #yes-btn {
            padding: 12px 35px;
            font-size: 18px;
        }
        #no-btn {
            padding: 10px 25px;
            font-size: 16px;
        }
    }
</style>

<script>
    const noBtn = document.getElementById('no-btn');
    const yesBtn = document.getElementById('yes-btn');
    const container = document.getElementById('game-container');

    // Khởi tạo vị trí ban đầu cho nút KHÔNG (nằm dưới nút CÓ một chút trên mobile hoặc bên cạnh trên PC)
    noBtn.style.left = '50%';
    noBtn.style.top = '75%';
    noBtn.style.transform = 'translateX(-50%)';

    const moveNoBtn = (e) => {
        const containerRect = container.getBoundingClientRect();
        const yesRect = yesBtn.getBoundingClientRect();
        const btnWidth = noBtn.offsetWidth;
        const btnHeight = noBtn.offsetHeight;

        let newX, newY;
        let isOverlapping = true;

        // Vòng lặp tìm vị trí mới cho đến khi không đè lên nút CÓ
        while (isOverlapping) {
            newX = Math.random() * (containerRect.width - btnWidth - 20) + 10;
            newY = Math.random() * (containerRect.height - btnHeight - 20) + 10;

            // Kiểm tra va chạm với nút CÓ (Yes)
            // Lấy tọa độ tương đối của nút Có trong container
            const yesLeft = yesRect.left - containerRect.left;
            const yesTop = yesRect.top - containerRect.top;
            
            // Tạo một vùng an toàn quanh nút CÓ (padding 50px)
            const safetyMargin = 60;
            
            const overlapX = newX + btnWidth > yesLeft - safetyMargin && newX < yesLeft + yesRect.width + safetyMargin;
            const overlapY = newY + btnHeight > yesTop - safetyMargin && newY < yesTop + yesRect.height + safetyMargin;

            if (!overlapX || !overlapY) {
                isOverlapping = false;
            }
        }

        noBtn.style.left = newX + 'px';
        noBtn.style.top = newY + 'px';
        noBtn.style.transform = 'none';
    };

    // Sự kiện cho cả PC và Mobile
    noBtn.addEventListener('mouseover', moveNoBtn);
    noBtn.addEventListener('touchstart', (e) => {
        e.preventDefault(); // Ngăn chặn click giả lập
        moveNoBtn();
    });
    
    noBtn.addEventListener('click', (e) => {
        e.preventDefault();
        alert('Hệ thống phát hiện nỗ lực gian lận! Vui lòng chọn lại.');
    });

    yesBtn.addEventListener('click', () => {
        alert('🎉 Chúc mừng! Bạn đã vượt qua bài kiểm tra độ trung thực của AI.');
        window.parent.location.reload(); 
    });
</script>
"""

# Render
components.html(game_html, height=650)

st.info("💡 Tip: AI sử dụng cảm biến tiệm cận để né tránh sự dối trá.")
