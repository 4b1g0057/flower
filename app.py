# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# --- 設定頁面資訊 ---
st.set_page_config(
    page_title="台灣路上常見花朵辨識",
    layout="centered"
)

# --- 自定義樣式 (可選) ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 標題區 ---
st.title("台灣路上常見花朵辨識")
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.caption("鬼針草(大花咸豐草)")
with col2: st.caption("雞蛋花")
with col3: st.caption("日日春")
with col4: st.caption("馬櫻丹")

st.markdown("---")

# --- 1. 載入模型 ---
@st.cache_resource
def load_model():
    model_path = 'best.pt' 
    if not os.path.exists(model_path):
        st.error(" 找不到模型檔案。請先執行完成訓練。")
        return None
    return YOLO(model_path)

model = load_model()

conf_threshold = 0.5


flower_dict = {
    'Bidens': '鬼針草(大花咸豐草)',
    'Frangipani': '雞蛋花',
    'Periwinkle': '日日春',
    'Lantana': '馬櫻丹'
}

# --- 3. 圖片上傳與推論 ---
uploaded_file = st.file_uploader("請上傳一張照片...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 顯示原圖
    image = Image.open(uploaded_file)
    st.image(image, caption='原始圖片', use_container_width=True)

    if model and st.button('開始辨識'):
        with st.spinner('正在分析...'):
            try:
                # 進行預測
                results = model.predict(image, conf=conf_threshold)
                
                # 繪製方框並取得圖片陣列
                res_plotted = results[0].plot()
                res_image = Image.fromarray(res_plotted[..., ::-1]) # BGR 轉 RGB
                
                st.success("辨識完成！")
                st.image(res_image, caption='辨識結果', use_container_width=True)
                
                # 顯示詳細結果
                st.subheader("偵測報告")
                
                detected_boxes = results[0].boxes
                if len(detected_boxes) == 0:
                    st.warning("畫面中未偵測到指定花朵，請嘗試調整拍攝角度或降低信心門檻值。")
                else:
                    for box in detected_boxes:
                        cls_id = int(box.cls[0])
                        eng_name = model.names[cls_id]
                        chn_name = flower_dict.get(eng_name, eng_name)
                        conf = float(box.conf[0])
                        
                        # 顯示單個偵測結果
                        st.markdown(f"""
                        <div style="background-color: white; padding: 10px; border-radius: 5px; margin-bottom: 5px; border-left: 5px solid #4CAF50;">
                            <b>{chn_name}</b> <small>({eng_name})</small> 
                            <span style="float:right;">信心指數: <b>{conf:.2f}</b></span>
                        </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"發生錯誤: {e}")
