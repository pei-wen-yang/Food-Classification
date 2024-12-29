import requests
from linebot import LineBotApi
from linebot.models import TextSendMessage, ImageSendMessage

# 上傳圖片至 Imgur
def upload_image_to_imgur(file_path):
    # Imgur API 設定
    CLIENT_ID = '261ad6e8e8e0db4'
    
    url = "https://api.imgur.com/3/upload"
    headers = {
        'Authorization': f'Client-ID {CLIENT_ID}',
    }
    with open(file_path, 'rb') as img:
        response = requests.post(url, headers=headers, files={'image': img})
    data = response.json()
    if data['success']:
        return data['data']['link']
    else:
        raise Exception("上傳失敗，請檢查 API 回應")


def send_line_message(message,image_url):
    #初始化 Line Bot API
    CHANNEL_ACCESS_TOKEN = 'oocldWlJxaWyV6xc9wPqDtvMy77xon7vFs+pPZ2QHngz1psdb5JZ2JX9CVf6LdbnfKuU5ArA3SACUI30mVUR1unFwwF3FEEM5fCtdyhU4Ds9eTNWEJ9e4An0oXaB4OQG0wFiW/iNgQGYKa1639AN7wdB04t89/1O/w1cDnyilFU='
    CHANNEL_SECRET = '6f03f728e9f2780dfbdf3d76e299e3da'
    line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)

    
    USER_ID = 'U3cf2c9fa0763762b469d2d19dc6941d7'  # 替換為目標使用者的 USER_ID
    
    """Send a text message via LINE"""
    try:
        image_message = ImageSendMessage(
            original_content_url=image_url,
            preview_image_url=image_url
        )
        line_bot_api.push_message(USER_ID, TextSendMessage(text=message))
        line_bot_api.push_message(USER_ID, image_message)
    except Exception as e:
        print(f"Failed to send LINE text message: {e}")

