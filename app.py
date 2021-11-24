from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_cors import CORS, cross_origin
import json
from flask_migrate import Migrate
# from sqlalchemy.dialects.postgresql import UUID

# from ai.ai_diagnosis import ai_diagnosis

# from keras.models import load_model
# model = load_model('model.h5')

app = Flask(__name__)
CORS(app)
app.config[ "SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:password@localhost:5432/testdb3"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app) # not sure if app should be in parameter or not

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
    __tablename__ = 'patient'
    # PatientId = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    address = db.Column(db.String(500))
    contact = db.Column(db.Integer)
    reports = db.relationship("Report", backref="patient", lazy=True)

    # def __repr__(self ) -> str:
    #    return f"{self.SerialNo} - {self.title}"


class Doctor(User):
    __tablename__ = 'doctor'
    # DoctorId = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    specialisation = db.Column(db.String(200))
    contact = db.Column(db.Integer)
    qualification = db.Column(db.String(200))
    hospital_id = db.Column(db.Integer, db.ForeignKey("hospital.id"), nullable=False)
    reports = db.relationship("Report", backref="doctor", lazy=True)


class Admin(User):
    __tablename__ = 'admin'
    name = db.Column(db.String(200), nullable=False)


class Hospital(db.Model):
    __tablename__ = 'hospital'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    address = db.Column(db.String(500))
    noOfDoctors = db.Column(db.Integer)
    contact = db.Column(db.Integer)
    doctors = db.relationship("Doctor", backref="hospital", lazy=True)


class Report(db.Model):
    __tablename__ = 'report'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey("doctor.id"), nullable=False)
    ai_diagnosis = db.Column(db.String(500))
    doctor_comment = db.Column(db.String(500))
    medication = db.Column(db.String(500))
    date_created = db.Column(db.DateTime, default=datetime.utcnow)



# @app.route("/signup", methods=["GET", "POST"])
# def signup():
#     if request.method == "GET":
#         render_template("signup.html")
#     if request.method == "POST":
#         email = request.form.get("email")
#         Username = request.form.get("username")
#         password = request.form.get("password")
#         user = User.query.filter_by(email=email).first()
#         # if email already in database
#         if user:
#             return "User email already exists"
#         # if username already in database
#         user2 = User.query.filter_by(UserName=Username).first()
#         if user2:
#             return "Username already exists"


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
    #resp = ai_diagnosis(file, jsondata)
    # resp = {'Model 1': 'Depression', 'Model 2': 'Depression', 'Model 3': 'Depression', 'Model 4': 'Depression', 'Model 5': 'Depression', 'Model 6': 'Depression', 'Model 7': 'Depression', 'Model 8': 'Depression', 'Model 9': 'Depression', 'Model 10': 'Depression', 'Model 11': 'Depression', 'Model 12': 'Depression'} 
    #print(resp)
    response = {"message": "Successfully Uploaded"}

    return response, 200

@app.route("/signup", methods=['POST'])
@cross_origin()
def signup():
    json_data = request.get_json()
    _role = json_data["role"]
    _username = json_data["username"]
    _email = json_data["email"]
    _password = json_data["password"]
    _id = json_data["id"]
    h_id = json_data["hid"]
    if _role and _email and _username and _password:
        # new_user = InfoModel(name=name, age=age)
        # db.session.add(new_user)
        # db.session.commit()
        if _role == 'Admin':
            new_admin = Admin()
            new_admin.name = _username
            new_admin.email = _email
            new_admin.password = _password
            db.session.add(new_admin)
            db.session.commit()

        if _role == 'Doctor':
            new_doctor = Doctor()
            new_doctor.id = _id
            new_doctor.hospital_id = h_id
            new_doctor.name = _username
            new_doctor.email = _email
            new_doctor.password = _password
            db.session.add(new_doctor)
            db.session.commit()
            
        if _role == 'Patient':
            new_patient = Patient()
            new_patient.name = _username
            new_patient.email = _email
            new_patient.password = _password
            db.session.add(new_patient)
            db.session.commit()
        
        response = {"message": "Successfully Uploaded"}
        return response, 200
    
    else:
        response = {"message": "Failed"}
        return response, 400
        
if __name__ == "__main__":
    app.run(debug=True)
