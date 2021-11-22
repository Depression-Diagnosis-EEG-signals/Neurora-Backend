from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_cors import CORS, cross_origin
import json
import pandas as pd

# from keras.models import load_model
# model = load_model('model.h5')

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

    return response, 200


if __name__ == "__main__":
    app.run(debug=True)
