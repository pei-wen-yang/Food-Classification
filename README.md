## 112453009 楊沛雯
## 硬體環境需求

`1.樹莓派4 (respberry  4)` 

`2.USB相機`

`3.PIR紅外線感測器`

`4.HX711重量感測器`

[![kaggle](https://i.imgur.com/OeQeTlg.png)]()

## 軟體環境需求

`食物辨識資料集下載`

## 🔗 Food-classification 

[![kaggle](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*JSbnt_mxpFfkGtNtGbR40g.png)](https://www.kaggle.com/datasets/bjoernjostein/food-classification)

`訓練資料集`

🔗 Demo參考以下連結
https://www.kaggle.com/code/oussamab25/food-class


# 架設完成圖
[![Finish](https://i.imgur.com/G2z4Z4w.jpg)]()



## 安裝指令
以下為專案運行所需的依賴模組及安裝指令：

### 必須的依賴套件
1. **RPi.GPIO**  
   用於控制樹莓派 GPIO 腳位的必要模組。  
   ```bash
   sudo apt install python3-rpi-lgpio
   ```

### 2. OpenCV
用於處理影像的功能。
```bash
pip install opencv-python
pip install opencv-python-headless
```

### 3. Picamera2
控制 Raspberry Pi 相機的模組。
```bash
pip install picamera2
sudo apt-get install -y libcamera-apps
```

### 4. LINE Bot SDK
用於透過 LINE 傳送影片訊息。
```bash
pip install line-bot-sdk
```

### 5. Requests
用於處理 HTTP 請求，例如上傳影片至 Imgur 的功能。
```bash
pip install requests
```

### 6. Pytz
處理時區所需的模組。
```bash
pip install pytz
```

### 備註


#### 相機測試建議
在開始正式運作前，可以使用以下指令來測試相機是否正常工作：  
```bash
libcamera-hello
```
#### 啟動網路功能
確保樹莓派已連接到網路，並且可以成功訪問 LINE API 與 Imgur 的服務。 










