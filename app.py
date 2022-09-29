import pickle
from re import L
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# load saved model
with open('model.pkl' , 'rb') as f:
    xgb_model = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print('\n\n',data)
    l = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    new_data = pd.DataFrame(list(data.values())).T
    new_data.columns = l
    print('\n\n',new_data)
    #print(np.array(list(data.values())).reshape(-1,1))
    #new_data = np.array(list(data.values())).reshape(-1,1)
    output = xgb_model.predict(new_data)
    print('\n\n',output[0])
    return jsonify(str(output[0]))

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    l = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    final_data = pd.DataFrame(data).T
    final_data.columns = l
    print(final_data)
    output = xgb_model.predict(final_data)[0]
    print(output)
    return render_template("home.html",prediction_text="The predicted house price is {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)



    