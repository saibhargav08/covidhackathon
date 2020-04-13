from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import base64
app = Flask(__name__)
import cv2
import numpy as np
from keras.models import model_from_json
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True



@app.route('/upload')
def upload_file():
   return render_template('upload.html')
	
@app.route('/predict', methods = ['GET','POST'])
def upload_image():
   if request.method == 'POST':
   	image = request.files['file']
   	img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
   	print(img)
   	json_file = open('./covid_hackathon.json', 'r')
   	loaded_model_json = json_file.read()
   	json_file.close()
   	loaded_model = model_from_json(loaded_model_json)
   	loaded_model.load_weights("./model_weights_res50.h5")
   	predict_image = [cv2.resize(img, (224, 224)).astype('float32')]
   	predict_image = np.array(predict_image)
   	pred = loaded_model.predict(predict_image)
   if np.argmax(pred[0], axis=None)==0:
   	covid_result = 'Negative'
   else:
   	covid_result = 'Positive'
   image_string = base64.b64encode(image.read()).decode('ascii')
   return render_template('prediction.html', result=image_string, covid_result=covid_result)
 
@app.route('/')
def home():
	return render_template('covid_detector.html')
		
if __name__ == '__main__':
   app.run(debug = True)
