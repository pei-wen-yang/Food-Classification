from gpiozero import MotionSensor
from datetime import datetime
import os
import time


from LineMessage import send_line_message,upload_image_to_imgur
from IdentifyFood import identify_food_name
from Weight import GetWeight,initWeight

# PIR
pir = MotionSensor(26)
hx = initWeight()



try:
    while True:
        print("start")
        pir.wait_for_motion()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pic_filename = f"image{timestamp}.jpg"
        # Take picture
        print('start')
        os.system(f"fswebcam -r 640x480 {pic_filename}")
        print('finish')
        
    
        
        # 上傳本地圖片並發送
        file_path = rf'/home/user/Final_Project/{pic_filename}' # 即時產生的圖片路徑
        
        # classfication image
        predicted_food_name = identify_food_name(file_path)
        food_weight = round(GetWeight(hx),2)
       
        image_url = upload_image_to_imgur(file_path)  # 上傳至 Imgur
        

        
        
        send_line_message(f"記錄日期:{timestamp}\n識別食物:{predicted_food_name}\n重量:{food_weight}g",image_url) # 發送圖片至 Line
        #send_line_image(image_path)
        # 加入延遲，避免短時間內重複觸發
        time.sleep(1)
except KeyboardInterrupt:
    print("Ending program")
    
    

                    
