from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_cors import CORS, cross_origin
import json
import pandas as pd
import numpy as np
import pickle
import pywt
from scipy import signal
from scipy.fftpack import fft, ifft
import statsmodels.api as sm
import antropy as ant
from scipy.stats import kurtosis
from scipy.stats import skew

from keras.models import load_model
model1 = load_model('Models/CNN_task.h5')
model2 = load_model('Models/CNN_EC.h5')
model3 = load_model('Models/CNN_EO.h5')
model4 = load_model('Models/CNN_LSTM_task.h5')
model5 = load_model('Models/CNN_LSTM_EC.h5')
model6 = load_model('Models/CNN_LSTM_EO.h5')
model7 = load_model('Models/CNN_LSTM_pywt1_task.h5')
model8 = load_model('Models/CNN_LSTM_pywt2_task.h5')
model9 = load_model('Models/CNN_LSTM_pywt3_task.h5')
model10 = load_model('Models/CNN_LSTM_pywt1_EC.h5')
model11 = load_model('Models/CNN_LSTM_pywt2_EC.h5')
model12 = load_model('Models/CNN_LSTM_pywt3_EC.h5')
model13 = load_model('Models/CNN_LSTM_pywt1_EO.h5')
model14 = load_model('Models/CNN_LSTM_pywt2_EO.h5')
model15 = load_model('Models/CNN_LSTM_pywt3_EO.h5')
model16 = load_model('Models/CNN_LSTM_FS_task.h5')
model17 = load_model('Models/CNN_LSTM_FS_EC.h5')
model18 = load_model('Models/CNN_LSTM_FS_EO.h5')
model19 = load_model('Models/ANN_task.h5')
model20 = load_model('Models/ANN_EC.h5')
model21 = load_model('Models/ANN_EO.h5')
model22 = pickle.load(open('Models/svm_task.sav', 'rb'))
model23 = pickle.load(open('Models/svm_fd_task.sav', 'rb'))
model24 = pickle.load(open('Models/svm_EC.sav', 'rb'))
model25 = pickle.load(open('Models/svm_fd_EC.sav', 'rb'))
model26 = pickle.load(open('Models/svm_EO.sav', 'rb'))
model27 = pickle.load(open('Models/svm_fd_EO.sav', 'rb'))
model28 = pickle.load(open('Models/dt_task.sav', 'rb'))
model29 = pickle.load(open('Models/dt_EC.sav', 'rb'))
model30 = pickle.load(open('Models/dt_EO.sav', 'rb'))
model31 = pickle.load(open('Models/rf_task.sav', 'rb'))
model32 = pickle.load(open('Models/rf_EC.sav', 'rb'))
model33 = pickle.load(open('Models/rf_EO.sav', 'rb'))
model34 = pickle.load(open('Models/KNN_task.sav', 'rb'))
model35 = pickle.load(open('Models/KNN_EC.sav', 'rb'))
model36 = pickle.load(open('Models/KNN_EO.sav', 'rb'))
app = Flask(__name__)
CORS(app)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///neurora.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# database models


class User(db.Model):
    __abstract__ = True
    id = db.Column(db.Integer, primary_key=True)
    #  username = db.Column(db.String(200), nullable=False, primary_key=True)
    age = db.Column(db.Integer)
    email = db.Column(db.String(200))
    password = db.Column(db.String(200))
    gender = db.Column(db.String(10))


class Patient(User):
    # PatientId = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    address = db.Column(db.String(500))
    contact = db.Column(db.Integer)
    reports = db.relationship("Report", backref="patient", lazy=True)

    # def __repr__(self ) -> str:
    #    return f"{self.SerialNo} - {self.title}"


class Doctor(User):
    # DoctorId = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    specialisation = db.Column(db.String(200))
    contact = db.Column(db.Integer)
    qualification = db.Column(db.String(200))
    hospital_id = db.Column(db.Integer, db.ForeignKey("hospital.id"), nullable=False)
    reports = db.relationship("Report", backref="doctor", lazy=True)


class Admin(User):
    name = db.Column(db.String(200), nullable=False)


class Hospital(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    address = db.Column(db.String(500))
    noOfDoctors = db.Column(db.Integer)
    contact = db.Column(db.Integer)
    doctors = db.relationship("Doctor", backref="hospital", lazy=True)


class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey("doctor.id"), nullable=False)
    ai_diagnosis = db.Column(db.String(500))
    doctor_comment = db.Column(db.String(500))
    medication = db.Column(db.String(500))
    date_created = db.Column(db.DateTime, default=datetime.utcnow)


@app.route("/predict", methods=["POST"])
def predict():
    imagefile = request.files["imagefile"]
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]

    classification = ""
    return render_template("index.html", prediction=classification)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        render_template("signup.html")
    if request.method == "POST":
        email = request.form.get("email")
        Username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(email=email).first()
        # if email already in database
        if user:
            return "User email already exists"
        # if username already in database
        user2 = User.query.filter_by(UserName=Username).first()
        if user2:
            return "Username already exists"


@app.route("/api", methods=["POST"])
@cross_origin()
def upload():
    try:
        file = request.files["file"]
    except:
        response = {"message": "Failed"}
        return response, 400
    data = request.form.to_dict()["data"]
    jsondata = json.loads(data)
    # print(jsondata)
    df = pd.read_csv(file)
    # print(df)
    response = {"message": "Successfully Uploaded"}
    if(jsondata['type']=="Task"):
    {
        df=df.drop(["Time"],axis=1)
        x_test=df.to_numpy()
        k=x_test
        x_test=x_test.reshape(1,2560,19)
        ka1=model1.predict(x_test)
        ka2=model4.predict(x_test)
        a1=np.empty((326,19),float)
        a2=np.empty((166,19),float)
        a3=np.empty((86,19),float)
        for i in range(0,19):
            coeffs = pywt.wavedec(k[:,i], 'db4',level=6)
            py1=coeffs[4]
            py2=coeffs[3]
            py3=coeffs[2]
            a1[:,i]=py1
            a2[:,i]=py2
            a3[:,i]=py3
        a1=a1.reshape(1,326,19)
        a2=a2.reshape(1,166,19)
        a3=a3.reshape(1,86,19)
        ka3=model7.predict(a1)
        ka4=model8.predict(a2)
        ka5=model9.predict(a3)
        ka=np.empty((50,19),float)
        for i in range(0,19):
            f, Pxx_den = signal.welch(k[:,i], 256, nperseg=1024)
            kafka=np.empty(1,float)
            kafka=np.mean(Pxx_den)
            kafka=np.append(kafka,[np.median(Pxx_den)])
            kafka=np.append(kafka,[np.std(Pxx_den)])
            kafka=np.append(kafka,[np.var(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)])
            kafka=np.append(kafka,[np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)-np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.mean(np.square(Pxx_den))])
            kafka=np.append(kafka,[ant.perm_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.spectral_entropy(Pxx_den, sf=256, method='welch', normalize=True)])
            kafka=np.append(kafka,[ant.svd_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.hjorth_params(Pxx_den)])
            kafka=np.append(kafka,[ant.petrosian_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.katz_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.higuchi_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.detrended_fluctuation(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den,fisher=False)])
            kafka=np.append(kafka,[skew(Pxx_den)])
            f, Pxx_den = signal.welch(k[:,i], 256, 'flattop',nperseg=1024, scaling='spectrum')
            Pxx_den=np.sqrt(Pxx_den)
            kafka=np.append(kafka,[np.mean(Pxx_den)])
            kafka=np.append(kafka,[np.median(Pxx_den)])
            kafka=np.append(kafka,[np.std(Pxx_den)])
            kafka=np.append(kafka,[np.var(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)])
            kafka=np.append(kafka,[np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)-np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.mean(np.square(Pxx_den))])
            kafka=np.append(kafka,[ant.perm_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.spectral_entropy(Pxx_den, sf=256, method='welch', normalize=True)])
            kafka=np.append(kafka,[ant.svd_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.hjorth_params(Pxx_den)])
            kafka=np.append(kafka,[ant.petrosian_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.katz_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.higuchi_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.detrended_fluctuation(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den,fisher=False)])
            kafka=np.append(kafka,[skew(Pxx_den)])
            rho, sigma = sm.regression.linear_model.burg(k[:,i], order=4)
            kafka=np.append(kafka,[sigma])
            kafka=np.append(kafka,rho)
            rho, sigma = sm.regression.yule_walker(k[:,i], order=4,method='mle')
            kafka=np.append(kafka,[sigma])
            kafka=np.append(kafka,rho)
            ka[:,i]=kafka
        ka=ka.reshape(1,50,19)
        ka6=model16.predict(ka)
        kafka=np.empty(0,float)
        for i in range(0,19):
            Pxx_den=k[:,i]
            kafka=np.append(kafka,[np.mean(Pxx_den)])
            kafka=np.append(kafka,[np.median(Pxx_den)])
            kafka=np.append(kafka,[np.std(Pxx_den)])
            kafka=np.append(kafka,[np.var(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)])
            kafka=np.append(kafka,[np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)-np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.mean(np.square(Pxx_den))])
            kafka=np.append(kafka,[ant.perm_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.spectral_entropy(Pxx_den, sf=256, method='welch', normalize=True)])
            kafka=np.append(kafka,[ant.svd_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.hjorth_params(Pxx_den)])
            kafka=np.append(kafka,[ant.petrosian_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.katz_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.higuchi_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.detrended_fluctuation(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den,fisher=False)])
            kafka=np.append(kafka,[skew(Pxx_den)])
        kafka=kafka.reshape(1,380)
        ka7=model19.predict(kafka)
        kafka=np.empty(0,float)
        for i in range(0,19):
            Pxx_den=k[:,i]
            kafka=np.append(kafka,[ant.perm_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.spectral_entropy(Pxx_den, sf=256, method='welch', normalize=True)])
            kafka=np.append(kafka,[ant.svd_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.hjorth_params(Pxx_den)])
        kafka=kafka.reshape(1,-1)
        ka8=model22.predict(kafka)
        kafka=np.empty(0,float)
        for i in range(0,19):
            Pxx_den=k[:,i]
            kafka=np.append(kafka,[ant.petrosian_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.katz_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.higuchi_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.detrended_fluctuation(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den,fisher=False)])
            kafka=np.append(kafka,[skew(Pxx_den)])
        kafka=kafka.reshape(1,-1)
        ka9=model23.predict(kafka)
        kafka=np.empty(0,float)
        for i in range(0,19):
            Pxx_den=k[:,i]
            kafka=np.append(kafka,[np.mean(Pxx_den)])
            kafka=np.append(kafka,[np.median(Pxx_den)])
            kafka=np.append(kafka,[np.std(Pxx_den)])
            kafka=np.append(kafka,[np.var(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)])
            kafka=np.append(kafka,[np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)-np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.mean(np.square(Pxx_den))])
            kafka=np.append(kafka,[ant.perm_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.spectral_entropy(Pxx_den, sf=256, method='welch', normalize=True)])
            kafka=np.append(kafka,[ant.svd_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.hjorth_params(Pxx_den)])
            kafka=np.append(kafka,[ant.petrosian_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.katz_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.higuchi_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.detrended_fluctuation(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den,fisher=False)])
            kafka=np.append(kafka,[skew(Pxx_den)])
        kafka=kafka.reshape(1,-1)
        ka10=model28.predict(kafka)
        ka11=model31.predict(kafka)
        ka12=model34.predict(kafka)
        ka1=ka1.argmax(axis=1)
        ka2=ka2.argmax(axis=1)
        ka3=ka3.argmax(axis=1)
        ka4=ka4.argmax(axis=1)
        ka5=ka5.argmax(axis=1)
        ka6=ka6.argmax(axis=1)
        ka7=ka7.argmax(axis=1)
        if(ka1[0]==0):
            response["Model 1"]="Healthy"
        else:
            response["Model 1"]="Depression"
        if(ka2[0]==0):
            response["Model 2"]="Healthy"
        else:
            response["Model 2"]="Depression"
        if(ka3[0]==0):
            response["Model 3"]="Healthy"
        else:
            response["Model 3"]="Depression"
        if(ka4[0]==0):
            response["Model 4"]="Healthy"
        else
            response["Model 4"]="Depression"
        if(ka5[0]==0):
            response["Model 5"]="Healthy"
        else:
            response["Model 5"]="Depression"
        if(ka6[0]==0):
            response["Model 6"]="Healthy"
        else:
            response["Model 6"]="Depression"
        if(ka7[0]==0):
            response["Model 7"]="Healthy"
        else:
            response["Model 7"]="Depression"
        if(ka8[0]==0):
            response["Model 8"]="Healthy"
        else:
            response["Model 8"]="Depression"
        if(ka9[0]==0):
            response["Model 9"]="Healthy"
        else:
            response["Model 9"]="Depression"
        if(ka10[0]==0):
            response["Model 10"]="Healthy"
        else:
            response["Model 10"]="Depression"
        if(ka11[0]==0):
            response["Model 11"]="Healthy"
        else:
            response["Model 11"]="Depression"
        if(ka12[0]==0):
            response["Model 12"]="Healthy"
        else:
            response["Model 12"]="Depression"
        
    }
    elif(jsondata['type']=="Eyes Closed"):
    {
        df=df.drop(["Time"],axis=1)
        x_test=df.to_numpy()
        k=x_test
        x_test=x_test.reshape(1,2560,19)
        ka1=model2.predict(x_test)
        ka2=model5.predict(x_test)
        a1=np.empty((326,19),float)
        a2=np.empty((166,19),float)
        a3=np.empty((86,19),float)
        for i in range(0,19):
            coeffs = pywt.wavedec(k[:,i], 'db4',level=6)
            py1=coeffs[4]
            py2=coeffs[3]
            py3=coeffs[2]
            a1[:,i]=py1
            a2[:,i]=py2
            a3[:,i]=py3
        a1=a1.reshape(1,326,19)
        a2=a2.reshape(1,166,19)
        a3=a3.reshape(1,86,19)
        ka3=model10.predict(a1)
        ka4=model11.predict(a2)
        ka5=model12.predict(a3)
        ka=np.empty((50,19),float)
        for i in range(0,19):
            f, Pxx_den = signal.welch(k[:,i], 256, nperseg=1024)
            kafka=np.empty(1,float)
            kafka=np.mean(Pxx_den)
            kafka=np.append(kafka,[np.median(Pxx_den)])
            kafka=np.append(kafka,[np.std(Pxx_den)])
            kafka=np.append(kafka,[np.var(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)])
            kafka=np.append(kafka,[np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)-np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.mean(np.square(Pxx_den))])
            kafka=np.append(kafka,[ant.perm_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.spectral_entropy(Pxx_den, sf=256, method='welch', normalize=True)])
            kafka=np.append(kafka,[ant.svd_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.hjorth_params(Pxx_den)])
            kafka=np.append(kafka,[ant.petrosian_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.katz_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.higuchi_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.detrended_fluctuation(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den,fisher=False)])
            kafka=np.append(kafka,[skew(Pxx_den)])
            f, Pxx_den = signal.welch(k[:,i], 256, 'flattop',nperseg=1024, scaling='spectrum')
            Pxx_den=np.sqrt(Pxx_den)
            kafka=np.append(kafka,[np.mean(Pxx_den)])
            kafka=np.append(kafka,[np.median(Pxx_den)])
            kafka=np.append(kafka,[np.std(Pxx_den)])
            kafka=np.append(kafka,[np.var(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)])
            kafka=np.append(kafka,[np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)-np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.mean(np.square(Pxx_den))])
            kafka=np.append(kafka,[ant.perm_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.spectral_entropy(Pxx_den, sf=256, method='welch', normalize=True)])
            kafka=np.append(kafka,[ant.svd_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.hjorth_params(Pxx_den)])
            kafka=np.append(kafka,[ant.petrosian_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.katz_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.higuchi_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.detrended_fluctuation(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den,fisher=False)])
            kafka=np.append(kafka,[skew(Pxx_den)])
            rho, sigma = sm.regression.linear_model.burg(k[:,i], order=4)
            kafka=np.append(kafka,[sigma])
            kafka=np.append(kafka,rho)
            rho, sigma = sm.regression.yule_walker(k[:,i], order=4,method='mle')
            kafka=np.append(kafka,[sigma])
            kafka=np.append(kafka,rho)
            ka[:,i]=kafka
        ka=ka.reshape(1,50,19)
        ka6=model17.predict(ka)
        kafka=np.empty(0,float)
        for i in range(0,19):
            Pxx_den=k[:,i]
            kafka=np.append(kafka,[np.mean(Pxx_den)])
            kafka=np.append(kafka,[np.median(Pxx_den)])
            kafka=np.append(kafka,[np.std(Pxx_den)])
            kafka=np.append(kafka,[np.var(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)])
            kafka=np.append(kafka,[np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)-np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.mean(np.square(Pxx_den))])
            kafka=np.append(kafka,[ant.perm_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.spectral_entropy(Pxx_den, sf=256, method='welch', normalize=True)])
            kafka=np.append(kafka,[ant.svd_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.hjorth_params(Pxx_den)])
            kafka=np.append(kafka,[ant.petrosian_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.katz_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.higuchi_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.detrended_fluctuation(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den,fisher=False)])
            kafka=np.append(kafka,[skew(Pxx_den)])
        kafka=kafka.reshape(1,380)
        ka7=model20.predict(kafka)
        kafka=np.empty(0,float)
        for i in range(0,19):
            Pxx_den=k[:,i]
            kafka=np.append(kafka,[ant.perm_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.spectral_entropy(Pxx_den, sf=256, method='welch', normalize=True)])
            kafka=np.append(kafka,[ant.svd_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.hjorth_params(Pxx_den)])
        kafka=kafka.reshape(1,-1)
        ka8=model24.predict(kafka)
        kafka=np.empty(0,float)
        for i in range(0,19):
            Pxx_den=k[:,i]
            kafka=np.append(kafka,[ant.petrosian_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.katz_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.higuchi_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.detrended_fluctuation(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den,fisher=False)])
            kafka=np.append(kafka,[skew(Pxx_den)])
        kafka=kafka.reshape(1,-1)
        ka9=model25.predict(kafka)
        kafka=np.empty(0,float)
        for i in range(0,19):
            Pxx_den=k[:,i]
            kafka=np.append(kafka,[np.mean(Pxx_den)])
            kafka=np.append(kafka,[np.median(Pxx_den)])
            kafka=np.append(kafka,[np.std(Pxx_den)])
            kafka=np.append(kafka,[np.var(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)])
            kafka=np.append(kafka,[np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)-np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.mean(np.square(Pxx_den))])
            kafka=np.append(kafka,[ant.perm_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.spectral_entropy(Pxx_den, sf=256, method='welch', normalize=True)])
            kafka=np.append(kafka,[ant.svd_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.hjorth_params(Pxx_den)])
            kafka=np.append(kafka,[ant.petrosian_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.katz_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.higuchi_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.detrended_fluctuation(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den,fisher=False)])
            kafka=np.append(kafka,[skew(Pxx_den)])
        kafka=kafka.reshape(1,-1)
        ka10=model29.predict(kafka)
        ka11=model32.predict(kafka)
        ka12=model35.predict(kafka)
        ka1=ka1.argmax(axis=1)
        ka2=ka2.argmax(axis=1)
        ka3=ka3.argmax(axis=1)
        ka4=ka4.argmax(axis=1)
        ka5=ka5.argmax(axis=1)
        ka6=ka6.argmax(axis=1)
        ka7=ka7.argmax(axis=1)
        if(ka1[0]==0):
            response["Model 1"]="Healthy"
        else:
            response["Model 1"]="Depression"
        if(ka2[0]==0):
            response["Model 2"]="Healthy"
        else:
            response["Model 2"]="Depression"
        if(ka3[0]==0):
            response["Model 3"]="Healthy"
        else:
            response["Model 3"]="Depression"
        if(ka4[0]==0):
            response["Model 4"]="Healthy"
        else
            response["Model 4"]="Depression"
        if(ka5[0]==0):
            response["Model 5"]="Healthy"
        else:
            response["Model 5"]="Depression"
        if(ka6[0]==0):
            response["Model 6"]="Healthy"
        else:
            response["Model 6"]="Depression"
        if(ka7[0]==0):
            response["Model 7"]="Healthy"
        else:
            response["Model 7"]="Depression"
        if(ka8[0]==0):
            response["Model 8"]="Healthy"
        else:
            response["Model 8"]="Depression"
        if(ka9[0]==0):
            response["Model 9"]="Healthy"
        else:
            response["Model 9"]="Depression"
        if(ka10[0]==0):
            response["Model 10"]="Healthy"
        else:
            response["Model 10"]="Depression"
        if(ka11[0]==0):
            response["Model 11"]="Healthy"
        else:
            response["Model 11"]="Depression"
        if(ka12[0]==0):
            response["Model 12"]="Healthy"
        else:
            response["Model 12"]="Depression"
        
    }
    elif(jsondata['type']=="Eyes Opened"):
    {
        df=df.drop(["Time"],axis=1)
        x_test=df.to_numpy()
        k=x_test
        x_test=x_test.reshape(1,2560,19)
        ka1=model3.predict(x_test)
        ka2=model6.predict(x_test)
        a1=np.empty((326,19),float)
        a2=np.empty((166,19),float)
        a3=np.empty((86,19),float)
        for i in range(0,19):
            coeffs = pywt.wavedec(k[:,i], 'db4',level=6)
            py1=coeffs[4]
            py2=coeffs[3]
            py3=coeffs[2]
            a1[:,i]=py1
            a2[:,i]=py2
            a3[:,i]=py3
        a1=a1.reshape(1,326,19)
        a2=a2.reshape(1,166,19)
        a3=a3.reshape(1,86,19)
        ka3=model13.predict(a1)
        ka4=model14.predict(a2)
        ka5=model15.predict(a3)
        ka=np.empty((50,19),float)
        for i in range(0,19):
            f, Pxx_den = signal.welch(k[:,i], 256, nperseg=1024)
            kafka=np.empty(1,float)
            kafka=np.mean(Pxx_den)
            kafka=np.append(kafka,[np.median(Pxx_den)])
            kafka=np.append(kafka,[np.std(Pxx_den)])
            kafka=np.append(kafka,[np.var(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)])
            kafka=np.append(kafka,[np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)-np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.mean(np.square(Pxx_den))])
            kafka=np.append(kafka,[ant.perm_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.spectral_entropy(Pxx_den, sf=256, method='welch', normalize=True)])
            kafka=np.append(kafka,[ant.svd_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.hjorth_params(Pxx_den)])
            kafka=np.append(kafka,[ant.petrosian_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.katz_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.higuchi_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.detrended_fluctuation(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den,fisher=False)])
            kafka=np.append(kafka,[skew(Pxx_den)])
            f, Pxx_den = signal.welch(k[:,i], 256, 'flattop',nperseg=1024, scaling='spectrum')
            Pxx_den=np.sqrt(Pxx_den)
            kafka=np.append(kafka,[np.mean(Pxx_den)])
            kafka=np.append(kafka,[np.median(Pxx_den)])
            kafka=np.append(kafka,[np.std(Pxx_den)])
            kafka=np.append(kafka,[np.var(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)])
            kafka=np.append(kafka,[np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)-np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.mean(np.square(Pxx_den))])
            kafka=np.append(kafka,[ant.perm_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.spectral_entropy(Pxx_den, sf=256, method='welch', normalize=True)])
            kafka=np.append(kafka,[ant.svd_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.hjorth_params(Pxx_den)])
            kafka=np.append(kafka,[ant.petrosian_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.katz_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.higuchi_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.detrended_fluctuation(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den,fisher=False)])
            kafka=np.append(kafka,[skew(Pxx_den)])
            rho, sigma = sm.regression.linear_model.burg(k[:,i], order=4)
            kafka=np.append(kafka,[sigma])
            kafka=np.append(kafka,rho)
            rho, sigma = sm.regression.yule_walker(k[:,i], order=4,method='mle')
            kafka=np.append(kafka,[sigma])
            kafka=np.append(kafka,rho)
            ka[:,i]=kafka
        ka=ka.reshape(1,50,19)
        ka6=model18.predict(ka)
        kafka=np.empty(0,float)
        for i in range(0,19):
            Pxx_den=k[:,i]
            kafka=np.append(kafka,[np.mean(Pxx_den)])
            kafka=np.append(kafka,[np.median(Pxx_den)])
            kafka=np.append(kafka,[np.std(Pxx_den)])
            kafka=np.append(kafka,[np.var(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)])
            kafka=np.append(kafka,[np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)-np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.mean(np.square(Pxx_den))])
            kafka=np.append(kafka,[ant.perm_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.spectral_entropy(Pxx_den, sf=256, method='welch', normalize=True)])
            kafka=np.append(kafka,[ant.svd_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.hjorth_params(Pxx_den)])
            kafka=np.append(kafka,[ant.petrosian_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.katz_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.higuchi_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.detrended_fluctuation(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den,fisher=False)])
            kafka=np.append(kafka,[skew(Pxx_den)])
        kafka=kafka.reshape(1,380)
        ka7=model21.predict(kafka)
        kafka=np.empty(0,float)
        for i in range(0,19):
            Pxx_den=k[:,i]
            kafka=np.append(kafka,[ant.perm_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.spectral_entropy(Pxx_den, sf=256, method='welch', normalize=True)])
            kafka=np.append(kafka,[ant.svd_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.hjorth_params(Pxx_den)])
        kafka=kafka.reshape(1,-1)
        ka8=model26.predict(kafka)
        kafka=np.empty(0,float)
        for i in range(0,19):
            Pxx_den=k[:,i]
            kafka=np.append(kafka,[ant.petrosian_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.katz_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.higuchi_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.detrended_fluctuation(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den,fisher=False)])
            kafka=np.append(kafka,[skew(Pxx_den)])
        kafka=kafka.reshape(1,-1)
        ka9=model27.predict(kafka)
        kafka=np.empty(0,float)
        for i in range(0,19):
            Pxx_den=k[:,i]
            kafka=np.append(kafka,[np.mean(Pxx_den)])
            kafka=np.append(kafka,[np.median(Pxx_den)])
            kafka=np.append(kafka,[np.std(Pxx_den)])
            kafka=np.append(kafka,[np.var(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)])
            kafka=np.append(kafka,[np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.amax(Pxx_den)-np.amin(Pxx_den)])
            kafka=np.append(kafka,[np.mean(np.square(Pxx_den))])
            kafka=np.append(kafka,[ant.perm_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.spectral_entropy(Pxx_den, sf=256, method='welch', normalize=True)])
            kafka=np.append(kafka,[ant.svd_entropy(Pxx_den, normalize=True)])
            kafka=np.append(kafka,[ant.hjorth_params(Pxx_den)])
            kafka=np.append(kafka,[ant.petrosian_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.katz_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.higuchi_fd(Pxx_den)])
            kafka=np.append(kafka,[ant.detrended_fluctuation(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den)])
            kafka=np.append(kafka,[kurtosis(Pxx_den,fisher=False)])
            kafka=np.append(kafka,[skew(Pxx_den)])
        kafka=kafka.reshape(1,-1)
        ka10=model30.predict(kafka)
        ka11=model33.predict(kafka)
        ka12=model36.predict(kafka)
        ka1=ka1.argmax(axis=1)
        ka2=ka2.argmax(axis=1)
        ka3=ka3.argmax(axis=1)
        ka4=ka4.argmax(axis=1)
        ka5=ka5.argmax(axis=1)
        ka6=ka6.argmax(axis=1)
        ka7=ka7.argmax(axis=1)
        if(ka1[0]==0):
            response["Model 1"]="Healthy"
        else:
            response["Model 1"]="Depression"
        if(ka2[0]==0):
            response["Model 2"]="Healthy"
        else:
            response["Model 2"]="Depression"
        if(ka3[0]==0):
            response["Model 3"]="Healthy"
        else:
            response["Model 3"]="Depression"
        if(ka4[0]==0):
            response["Model 4"]="Healthy"
        else
            response["Model 4"]="Depression"
        if(ka5[0]==0):
            response["Model 5"]="Healthy"
        else:
            response["Model 5"]="Depression"
        if(ka6[0]==0):
            response["Model 6"]="Healthy"
        else:
            response["Model 6"]="Depression"
        if(ka7[0]==0):
            response["Model 7"]="Healthy"
        else:
            response["Model 7"]="Depression"
        if(ka8[0]==0):
            response["Model 8"]="Healthy"
        else:
            response["Model 8"]="Depression"
        if(ka9[0]==0):
            response["Model 9"]="Healthy"
        else:
            response["Model 9"]="Depression"
        if(ka10[0]==0):
            response["Model 10"]="Healthy"
        else:
            response["Model 10"]="Depression"
        if(ka11[0]==0):
            response["Model 11"]="Healthy"
        else:
            response["Model 11"]="Depression"
        if(ka12[0]==0):
            response["Model 12"]="Healthy"
        else:
            response["Model 12"]="Depression"
        
    }
    return response, 200


if __name__ == "__main__":
    app.run(debug=True)
