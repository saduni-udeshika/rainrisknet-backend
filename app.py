import datetime
from net import Net
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from utils import (
    get_image_path,
    generate_access_token,
    decode_access_token,
    encrypt_password,
    verify_password,
    validate_access,
)
from assessment import Assessment
from db import get_user, add_user, add_assessment_data, get_assessment_data, get_all_assessment_data
from knowledge_graph import generate_knowledge_graph, visualize_graph

processor = Net()

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


def auth_middleware(req):
    authorization_header = req.headers["authorization"].split()
    if len(authorization_header) > 1:
        access_token = req.headers["authorization"].split()[1]
        if validate_access(access_token) is False:
            return {"status": "error", "message": "not_authorized"}
    return True


def get_email_from_request(request):
    access_token = request.headers["authorization"].split()[1]
    decoded_token = decode_access_token(access_token)
    return decoded_token["email"]


@app.route("/")
def hello_world():
    return "Server running"


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    access = auth_middleware(request)
    if access is not True:
        return access
    uploaded_file = request.files["file"]
    image_path = get_image_path(uploaded_file.filename)
    uploaded_file.save(image_path)
    prediction = processor.predict(image_path)
    return prediction.title()


@app.route("/percentage", methods=["POST"])
@cross_origin()
def damage_percentage():
    access = auth_middleware(request)
    if access is not True:
        return access

    email = get_email_from_request(request)

    uploaded_file1 = request.files["file1"]
    uploaded_file2 = request.files["file2"]

    disaster_type = request.form.get("disaster_type")
    location = request.form.get("location")
    date_str = request.form.get("date")

    if not date_str:
        return jsonify(error="Date field is missing or empty"), 400

    try:
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify(error="Invalid date format"), 400

    image1_path = get_image_path(uploaded_file1.filename)
    image2_path = get_image_path(uploaded_file2.filename)
    uploaded_file1.save(image1_path)
    uploaded_file2.save(image2_path)
    percentage = Assessment.damage_percentage(image1_path, image2_path)
    rounded_percentage = round(percentage, 2)  # Round off to 2 decimal places

    # Convert date to datetime with zero time component
    datetime_with_zero_time = datetime.datetime.combine(date, datetime.time())

    # Prepare the assessment report data
    assessment_data = {
        "disaster_type": disaster_type,
        "location": location,
        "date": datetime_with_zero_time,
        "damage_percentage": rounded_percentage,
        "email": email,
    }

    # Insert assessment report data into MongoDB collection
    add_assessment_data(assessment_data)

    return str(rounded_percentage)


@app.route("/signup", methods=["POST"])
@cross_origin()
def signup():
    json = request.json
    user = get_user(json["email"])
    if user is not None:
        return {"status": "error", "message": "user_exists"}
    encrypted_password = encrypt_password(json["password"])
    add_user({**json, "password": encrypted_password})
    access_token = generate_access_token(json["email"])
    return {
        "status": "success",
        "access_token": access_token,
    }


@app.route("/get-user", methods=["GET"])
@cross_origin()
def user():
    access_token = request.headers["authorization"].split()[1]
    if access_token is None or access_token == "null":
        return {"status": "error"}
    decoded_token = decode_access_token(access_token)
    user = get_user(decoded_token["email"])
    user_id = str(user["_id"])
    return {
        "status": "success",
        "message": "success",
        "user": {
            "_id": user_id,
            "email": user["email"],
            "firstName": user["firstName"],
            "lastName": user["lastName"],
        },
    }


@app.route("/login", methods=["POST"])
@cross_origin()
def login():
    email = request.json["email"]
    password = request.json["password"]
    user = get_user(email)
    if user is None:
        return {"status": "error", "message": "user_does_no_exist"}
    is_valid_password = verify_password(password, user["password"])
    if is_valid_password is False:
        return {"status": "error", "message": "invalid_credentials"}
    user_id = str(user["_id"])
    access_token = generate_access_token(email)
    return {
        "status": "success",
        "message": "logged_in",
        "access_token": access_token,
        "user": {
            "_id": user_id,
            "email": user["email"],
            "firstName": user["firstName"],
            "lastName": user["lastName"],
        },
    }


@app.route("/assessed", methods=["GET"])
@cross_origin()
def assessed():
    access = auth_middleware(request)
    if access is not True:
        return access

    email = get_email_from_request(request)
    data = get_assessment_data(email)
    return jsonify(data), 200

@app.route("/knowledge-graph", methods=["GET"])
@cross_origin()
def knowledge_graph():
    access = auth_middleware(request)
    if access is not True:
        return access

    assessment_data = get_all_assessment_data()
    graph = generate_knowledge_graph(assessment_data)
    visualize_graph(graph)

    return {"status": "success", "message": "knowledge_graph_generated"}