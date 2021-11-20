from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# from keras.models import load_model
# model = load_model('model.h5')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']= "sqlite:///neurora.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db= SQLAlchemy(app)

# database models 
class Patient(db.model):
   PatientId = db.Column(db.Integer, primary_key=True)
   firstName = db.Column(db.String(200), nullable=False)
   lastName = db.Column(db.String(200))
   address = db.Column(db.String(500))
   contact = db.Column(db.Integer)

   # def __repr__(self ) -> str:  
   #    return f"{self.SerialNo} - {self.title}"

class Doctor(db.model):
   DoctorId = db.Column(db.Integer, primary_key=True)
   firstName = db.Column(db.String(200), nullable=False)
   lastName = db.Column(db.String(200))
   specialisation = db.Column(db.String(200))
   contact = db.Column(db.Integer)
   qualification = db.Column(db.String(200))

class Hospital(db.model):
   hospitalId = db.Column(db.Integer, primary_key=True)
   hospitalName = db.Column(db.String(200), nullable=False)
   address = db.Column(db.String(500))
   noOfDoctors = db.Column(db.Integer)
   contact = db.Column(db.Integer)

class User(db.model):
   userName = db.Column(db.String(200), nullable=False)
   userAge = db.Column(db.Integer)
   userEmail = db.Column(db.String(200))
   userPassword = db.Column(db.String(200))
   userGender = db.Column(db.String(10))
   

class Admin(db.model):
   adminName = db.Column(db.String(200), nullable=False)

class PatientReport(db.model):
   PatientId = db.Column(db.Integer)
   DoctorId = db.Column(db.Integer)
   aiDiagnosis = db.Column(db.String(500))
   DoctorComment = db.Column(db.String(500))
   Medication = db.Column(db.String(500))
   date_created = db.Column(db.DateTime, deafault=datetime.utcnow)

@app.route('/predict', methods=['POST'])
# def predict():
#    features = [int(x) for x in request.form.values()]
#    final_features = [np.aray(features)]
#    prediction = model.predict(final_features)
#    return render_template('index.html', prediction_text='')
def predict():
   imagefile = request.files['imagefile']
   image_path = "./images/" + imagefile.filename
   imagefile.save(image_path)

   image = load_img(image_path, target_size=(224,224))
   image = img_to_array(image)
   image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
   image = preprocess_input(image)
   yhat = model.predict(image)
   label = decode_predictions(yhat)
   label = label[0][0]

   classification = ''
   return render_template('index.html', prediction = classification)

if __name__ == '__main__':
   app.run(Debug = True)

