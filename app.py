from operator import le
from flask import Flask, render_template, request, url_for
import joblib
import numpy as np

from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
model = joblib.load('model.lb')  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/project', methods=['POST', 'GET'])
def predict():
    prediction = None

    if request.method == 'POST':
        brand = request.form['brand']
        processor_brand = request.form['processor_brand']
        processor_name = request.form['processor_name']
        processor_gnrtn = request.form['processor_gnrtn']
        ram_gb = request.form['ram_gb']
        ram_type = request.form['ram_type']
        ssd = request.form['ssd']
        hdd = request.form['hdd']
        os = request.form['os']
        os_bit = request.form['os_bit']
        graphic_card_gb = request.form['graphic_card_gb']
        warranty = request.form['warranty']
        touchscreen = request.form['Touchscreen']
        msoffice = request.form['msoffice']
        rating = request.form['rating']

        # Encode categorical values using fixed categories
        encoders = {}
        category_maps = {
            'brand': ['ASUS', 'Lenovo', 'acer', 'Avita', 'HP', 'DELL', 'MSI', 'APPLE'],
            'processor_brand': ['Intel', 'AMD', 'M1'],
            'processor_name': ['Core i3', 'Core i5', 'Celeron Dual', 'Ryzen 5', 'Core i7', 'Core i9', 'M1', 'Pentium Quad', 'Ryzen 3', 'Ryzen 7', 'Ryzen 9'],
            'processor_gnrtn': ['10th', 'Not Available', '11th', '7th', '8th', '9th', '4th', '12th'],
            'ram_type': ['DDR4', 'LPDDR4', 'LPDDR4X', 'DDR5', 'DDR3', 'LPDDR3'],
            'os': ['Windows', 'DOS', 'Mac'],
            'os_bit': ['64-bit', '32-bit'],
            'warranty': ['No warranty', '1 year', '2 years', '3 years'],
            'Touchscreen': ['No', 'Yes'],
            'msoffice': ['No', 'Yes'],
            'ram_gb': ['4 GB', '8 GB', '16 GB', '32 GB'],
            'ssd': ['0 GB', '512 GB', '256 GB', '128 GB', '1024 GB', '2048 GB', '3072 GB'],
            'hdd': ['1024 GB', '0 GB', '512 GB', '2048 GB'],
            'graphic_card_gb': ['0 GB', '2 GB', '4 GB', '6 GB', '8 GB'],
            'rating': ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
        }

        for col, classes in category_maps.items():
            le = LabelEncoder()
            le.classes_ = np.array(classes)  # ✅ convert list to numpy array
            encoders[col] = le


        encoded_input = [
            encoders['brand'].transform([brand])[0],
            encoders['processor_brand'].transform([processor_brand])[0],
            encoders['processor_name'].transform([processor_name])[0],
            encoders['processor_gnrtn'].transform([processor_gnrtn])[0],
            encoders['ram_gb'].transform([ram_gb])[0],
            encoders['ram_type'].transform([ram_type])[0],
            encoders['ssd'].transform([ssd])[0],
            encoders['hdd'].transform([hdd])[0],
            encoders['os'].transform([os])[0],
            encoders['os_bit'].transform([os_bit])[0],
            encoders['graphic_card_gb'].transform([graphic_card_gb])[0],
            encoders['warranty'].transform([warranty])[0],
            encoders['Touchscreen'].transform([touchscreen])[0],
            encoders['msoffice'].transform([msoffice])[0],
            encoders['rating'].transform([rating])[0]
        ]

        pred = model.predict([encoded_input])[0]
        prediction = round(float(pred), 2)

    return render_template('project.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
