from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
df = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open('xgbModel.pkl', 'rb'))

@app.route('/') 
def index():
    locations = sorted(df['location'].unique())
    return render_template('index.html', locations = locations)


@app.route('/predict', methods = ['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('sqft')

    input = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    ans = pipe.predict(input)[0] 

    return str(np.round(ans, 2))

if __name__ == "__main__" :
    app.run(debug = True)

