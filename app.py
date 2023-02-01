import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle



#création de l'application
app = Flask(__name__)

#import de la sauvegarde du modèle
model = pickle.load(open('savedmodel.sav', 'rb'))


#création du service
@app.route('/')
def home():
	result = ' '
	return render_template('index.html', **locals())

#@app.route: signifies what to do when the browser hit the particular URL and for that,
# we write the function just below it. In this case, 
# when user will try to open our webpage it will render index.html.


#fonction predict
@app.route('/predict', methods=['POST', 'GET'])
def predict():
	sepal_length = float(request.form['sepal_length'])
	sepal_width = float(request.form['sepal_width'])
	petal_length = float(request.form['petal_length'])
	petal_width = float(request.form['petal_width'])

	result = model.predict([(sepal_length, sepal_width, petal_length, petal_width)])[0]
	#result = result.reshape(-1, 1)
	return render_template('index.html', **locals())

#lever une erreur
if __name__ == "__main__":
    app.run(debug=True)