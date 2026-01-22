from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)
model = joblib.load('model/houseTrain.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
prediction = None
if request.method == 'POST':
data = {
'OverallQual': int(request.form['OverallQual']),
'GrLivArea': float(request.form['GrLivArea']),
'TotalBsmtSF': float(request.form['TotalBsmtSF']),
'GarageCars': int(request.form['GarageCars']),
'YearBuilt': int(request.form['YearBuilt']),
'Neighborhood': request.form['Neighborhood']
}
df = pd.DataFrame([data])
prediction = round(model.predict(df)[0], 2)
return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
app.run(debug=True)
