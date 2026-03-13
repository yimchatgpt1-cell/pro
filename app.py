from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient
import os
import uuid

app = Flask(__name__)

# Initialize Client ตัวเดียวใช้ได้ทุกโมเดล
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="FLg8F46YDgsleKPyVCR1"
)

@app.route('/analyze-food', methods=['POST'])
def analyze_food():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image"}), 400
    
    file = request.files['image']
    temp_path = f"temp_{uuid.uuid4()}.jpg"
    file.save(temp_path)

    try:
        # --- ขั้นตอนที่ 1: Classification (ทายชื่อเมนู) ---
        # ใช้ model_id สำหรับ Menu Classification
        class_result = CLIENT.infer(temp_path, model_id="menu-a4fzp/3")
        menu_name = "ไม่ทราบชื่อเมนู"
        if class_result.get('predictions'):
            # ดึงชื่อคลาสที่ค่าความมั่นใจสูงสุด
            menu_name = class_result['predictions'][0]['class']

        # --- ขั้นตอนที่ 2: Object Detection (หาวัตถุดิบ) ---
        # ใช้ model_id สำหรับ Ingredient Detection (v33)
        detect_result = CLIENT.infer(temp_path, model_id="ingredient_detection-vouls/33")
        
        # กรองวัตถุดิบที่เจอ (เอาเฉพาะชื่อ ไม่เอาตัวซ้ำ)
        raw_preds = detect_result.get('predictions', [])
        ingredients = list(set([p['class'] for p in raw_preds if p['confidence'] > 0.25]))

        # --- ขั้นตอนที่ 3: ลบไฟล์และส่งผลกลับ ---
        os.remove(temp_path)

        return jsonify({
            "success": True,
            "menu": menu_name,          # ผลจากโมเดล 1
            "ingredients": ingredients,   # ผลจากโมเดล 2
            "raw_detection": raw_preds   # ส่งพิกัดไปเผื่อ Flutter ใช้วาดกรอบ
        })

    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)