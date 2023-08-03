from net import Net
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from utils import get_image_path, generate_access_token, decode_access_token, encrypt_password, verify_password, validate_access
from assessment import Assessment
from db import get_user, add_user

processor = Net()

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def auth_middleware(req):
    authorization_header = req.headers['authorization'].split()
    if len(authorization_header) > 1:
        access_token = req.headers['authorization'].split()[1]
        if validate_access(access_token) is False:
            return {
                "status": "error",
                "message": "not_authorized"
            }
    return True


@app.route('/')
def hello_world():
    return 'Server running'


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    access = auth_middleware(request)
    if access is not True:
        return access
    uploaded_file = request.files['file']
    image_path = get_image_path(uploaded_file.filename)
    uploaded_file.save(image_path)
    prediction = processor.predict(image_path)
    return prediction.title()


@app.route('/percentage', methods=['POST'])
@cross_origin()
def damage_percentage():
    access = auth_middleware(request)
    if access is not True:
        return access
    uploaded_file1 = request.files['file1']
    uploaded_file2 = request.files['file2']
    image1_path = get_image_path(uploaded_file1.filename)
    image2_path = get_image_path(uploaded_file2.filename)
    uploaded_file1.save(image1_path)
    uploaded_file2.save(image2_path)
    percentage = Assessment.damage_percentage(image1_path, image2_path)
    rounded_percentage = round(percentage, 2)  # Round off to 2 decimal places
    return str(rounded_percentage)


@app.route('/signup', methods=['POST'])
@cross_origin()
def signup():
    json = request.json
    user = get_user(json['email'])
    if user is not None:
        return {
            "status": "error",
            "message": "user_exists"
        }
    encrypted_password = encrypt_password(json['password'])
    add_user({**json, "password": encrypted_password})
    access_token = generate_access_token(json['email'])
    return {
        "status": "success",
        "access_token": access_token,
    }


@app.route('/get-user', methods=['GET'])
@cross_origin()
def user():
    access_token = request.headers['authorization'].split()[1]
    if access_token is None or access_token == "null":
        return {
            "status": "error"
        }
    decoded_token = decode_access_token(access_token)
    user = get_user(decoded_token['email'])
    user_id = str(user['_id'])
    return {
        "status": "success",
        "message": "success",
        "user": {"_id": user_id,
                 "email": user['email'],
                 "firstName": user['firstName'],
                 "lastName": user['lastName'],
                 }
    }


@app.route('/login', methods=['POST'])
@cross_origin()
def login():
    email = request.json['email']
    password = request.json['password']
    user = get_user(email)
    if user is None:
        return {
            "status": "error",
            "message": "user_does_no_exist"
        }
    is_valid_password = verify_password(password, user['password'])
    if is_valid_password is False:
        return {
            "status": "error",
            "message": "invalid_credentials"
        }
    user_id = str(user['_id'])
    access_token = generate_access_token(email)
    return {
        "status": "success",
        "message": "logged_in",
        "access_token": access_token,
        "user": {"_id": user_id,
                 "email": user['email'],
                 "firstName": user['firstName'],
                 "lastName": user['lastName'],
                 }
    }
