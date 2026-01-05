from ultralytics import YOLO
import os

def main():
    # --- 1. 設定檢查 ---
    # 確保 data.yaml 存在，不然程式會直接報錯
    yaml_file = 'data.yaml'
    if not os.path.exists(yaml_file):
        print(f" 錯誤：找不到 {yaml_file}！")
        print("請確認您已經把 Roboflow 下載的檔案解壓縮，並放在專案目錄下。")
        return

    # --- 2. 載入模型 ---
    # 使用 'yolov8n.pt' (Nano 版本)，速度最快，適合筆電或 Colab
    # 如果您電腦有強大顯卡 (RTX 3060以上)，可以改用 'yolov8s.pt' 提高準度

    model = YOLO('yolov8n.pt') 

    # --- 3. 開始訓練 ---
    print("開始訓練模型...")
    try:
        results = model.train(
            # 資料集設定檔 (Roboflow 下載的那個 yaml)
            data=yaml_file,
            
            # 訓練輪數：因為資料量少(增強後約50-60張)，建議跑 50~100 輪
            epochs=50,
            
            # 圖片大小：標準為 640
            imgsz=640,
            
            # 訓練批次：設為 16 比較穩定，如果記憶體不足可改 8 或 4
            batch=16,
            
            # 專案名稱：訓練結果會存在 runs/detect/這個名字/
            name='taiwan_flower_v1',
            
            # 提早停止：如果連續 10 輪沒有進步就自動停止，節省時間
            patience=10,
            
            # 裝置：自動偵測 GPU，如果沒有會用 CPU
            # 如果您確定要用 CPU，可填 device='cpu'
            device=0 if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu' 
        )
        
        print("\n✅ 訓練完成！")
        print(f"最佳模型權重已儲存於： runs/detect/taiwan_flower_v1/weights/best.pt")
        print("請將該 best.pt 複製到最外層目錄以便 app.py 使用。")

    except Exception as e:
        print(f"\n❌ 訓練過程中發生錯誤：{e}")

if __name__ == '__main__':
    main()