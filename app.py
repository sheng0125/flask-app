from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# 載入模型
model = load_model("crochet_model_20250107_153537.keras", compile=False)

label_mapping = {
    0: ['0', '鎖針'], 1: ['1', '引拔針'], 2: ['2', '短針'], 3: ['3', '中長針'], 4: ['4', '長針'],
    5: ['5', '長長針'], 6: ['6', '三卷長針'], 7: ['7', '短針加針'], 8: ['8', '短針三加針'], 9: ['9', '中長針加針'],
    10: ['10', '中長針三加針'], 11: ['11', '長針加針'], 12: ['12', '長針三加針'], 13: ['17', '中長針減針'], 14: ['18', '中長針減三針'],
    15: ['19', '長針減針'], 16: ['20', '長針減三針'], 17: ['21', '長長針減針'], 18: ['22', '長長針減三針'], 19: ['23', '爆米花針'],
    20: ['25', '長針兩針棗形針'], 21: ['26', '長針三針棗形針'], 22: ['27', '長針四針棗形針'], 23: ['28', '長針五針棗形針'], 24: ['29', '泡芙針'],
    25: ['30', '中長針四針棗形針']
}

# 預測函數
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return predicted_class, predictions

@app.route('/')
def index():
    return render_template('index.html', file=None, predicted_class=None, probabilities=[])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded', file=None, predicted_class=None, probabilities=[])

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected', file=None, predicted_class=None, probabilities=[])

    file_path = os.path.join('static/uploads', file.filename)
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    file.save(file_path)

    predicted_class, predictions = predict_image(file_path)

    predictions = [
        [label_mapping.get(idx, [f"{idx}", "未知標籤"]), f'{predictions[0][idx]:f}']
        for idx in range(len(predictions[0]))
    ]

    return render_template(
        'index.html',
        predicted_class=label_mapping.get(int(predicted_class), [f"{int(predicted_class)}", f"未知標籤"]),
        probabilities=predictions,
        file=file.filename
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
