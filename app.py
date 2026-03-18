from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient
import os
import uuid

app = Flask(__name__)

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="FLg8F46YDgsleKPyVCR1"
)

# 🗺️ ตารางแปลภาษาให้ตรงกับฐานข้อมูล
TRANSLATION_MAP = {
    'shrimp': 'กุ้ง',
    'squid': 'ปลาหมึก',
    'rice': 'ข้าว',
    'pork': 'หมู',
    'basil': 'ใบกะเพรา',
    'Minced pork': 'หมูสับ',    
    'noodles': 'เส้นใหญ่',
    'crispy pork': 'หมูกรอบ',
    'kale': 'คะน้า',
    'fried egg': 'ไข่ดาว',
}

@app.route('/analyze-food', methods=['POST'])
def analyze_food():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image"}), 400
    
    file = request.files['image']
    temp_path = f"temp_{uuid.uuid4()}.jpg"
    file.save(temp_path)

    try:
        # 1. เรียก Classification (หาชื่อเมนู)
        class_result = CLIENT.infer(temp_path, model_id="menu-a4fzp/3")
        eng_menu = class_result['predictions'][0]['class'] if class_result.get('predictions') else "ไม่มีเมนูอาหารนี้ในระบบ"
        if top_prediction['confidence'] > 0.7:
            menu_name = top_prediction['class']
        else:
            menu_name = "ไม่มีข้อมลูในระบบ"

        # 2. เรียก Object Detection (หาวัตถุดิบ)
        detect_result = CLIENT.infer(temp_path, model_id="ingredient_detection-vouls/35")
        raw_preds = detect_result.get('predictions', [])

        # 3. แปลผลลัพธ์เป็นภาษาไทยเพื่อไป Match กับ Firebase
        detected_thai = []
        for p in raw_preds:
            eng_name = p['class'].lower().strip()
            if p['confidence'] > 0.35: # กรองความมั่นใจ
                thai_name = TRANSLATION_MAP.get(eng_name)
                if thai_name and thai_name not in detected_thai:
                    detected_thai.append(thai_name)

        os.remove(temp_path)

        return jsonify({
            "success": True,
            "menu_name": eng_menu,      # ส่งไปแปลที่ Flutter ต่อ
            "ingredients_thai": detected_thai  # ส่งชื่อภาษาไทยไป Match Firebase ทันที
        })

    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
