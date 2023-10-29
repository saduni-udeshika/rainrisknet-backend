import datetime
from net import Net
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from utils import (
    get_image_path,
    generate_access_token,
    decode_access_token,
    encrypt_password,
    verify_password,
    validate_access,
)
from assessment import BuildingDamageAssessor
import io
import matplotlib.pyplot as plt
from db import get_user, add_user, add_assessment_data, get_assessment_data, add_flood_assessment_data, add_landslide_assessment_data, add_disaster_forecast_data, get_all_assessment_data, get_flood_assessment_data, get_disaster_forecast_reports, get_landslide_assessment_data, get_landslide_damage_data, get_disaster_forecast_data, add_ada_derana_scraped_data, add_floodlist_scraped_data, add_disaster_report_forms, get_floodlist_news, get_adaderana_news
import tensorflow as tf
import numpy as np
from flood_damage_predict_image import(preprocess_image)
from landslide_damage_predict_image import(dice_coefficient, IMG_HEIGHT, IMG_WIDTH)
from flask import jsonify
from disaster_forecast_predict import predict_disaster 
from knowledge_graph import generate_knowledge_graph, visualize_graph
from humanitarian_risk_assessment import fetch_data_from_web
from FloodList_Data_Scraping import fetch_floodlist_data_from_web
from building_damage_classifier import BuildingDamageClassifier
from building_damage_classifier import BuildingDamageClassifier

processor = Net()
# Initialize the BuildingDamageClassifier with the model path
building_damage_classifier = BuildingDamageClassifier('building_damage_model.h5')

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
    # Classify the uploaded images for building damage
    classification1 = building_damage_classifier.classify_image(image1_path)
    classification2 = building_damage_classifier.classify_image(image2_path)

    # Initialize the BuildingDamageAssessor class
    damage_assessor = BuildingDamageAssessor()

    # Calculate and return the building damage percentage
    rounded_percentage = damage_assessor.calculate_percentage(image1_path, image2_path)

    # Store the classification results in the database
    classification_data = {
        "image1_classification": classification1,
        "image2_classification": classification2,
    }
    assessment_data = {
        "disaster_type": disaster_type,
        "location": location,
        "date": date_str,
        "damage_percentage": rounded_percentage,
        "email": email,
        "classification_data": classification_data,
    }
    add_assessment_data(assessment_data)

    return jsonify({"damage_percentage": rounded_percentage})


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



@app.route('/predict-image-flood-environmental', methods=['POST'])
@cross_origin()
def predict_image():
    access = auth_middleware(request)
    if access is not True:
        return access
    
    uploaded_file = request.files['file']
    user_image_path = get_image_path(uploaded_file.filename)
    uploaded_file.save(user_image_path)
    
    date = request.form['date']
    location = request.form['location']
    disaster_type = request.form['disaster_type']
    
    # Load your trained model
    model = tf.keras.models.load_model('flood_damage_model.h5')
    
    # Preprocess the user-provided image
    input_arr = preprocess_image(user_image_path, target_size=(224, 224))
    
    # Predict the mask for the user-provided image
    user_prediction = model.predict(np.array([input_arr]))
    
    # Define a threshold value
    threshold = 0.35
    
    # Convert the predicted mask to binary
    binary_mask = np.where(user_prediction > threshold, 1, 0)
    
    # Calculate the percentage of flood damage
    percentage_damage = (np.sum(binary_mask) / np.prod(binary_mask.shape)) * 100
    formatted_percentage = "{:.2f}".format(percentage_damage)
  
    response = {
        "percentage_damage": formatted_percentage,
        "date": date,
        "location": location,
        "disaster_type": disaster_type
    }
    
    add_flood_assessment_data(response)

    #return jsonify(response), 200
    return str(response)


@app.route("/predict-flood-environmental", methods=["POST"])
@cross_origin()
def predict_flood():
    access = auth_middleware(request)
    if access is not True:
        return access
    uploaded_file = request.files["file"]
    image_path = get_image_path(uploaded_file.filename)
    uploaded_file.save(image_path)
    prediction = processor.predict_flood(image_path)
    return prediction.title()


@app.route("/assessed-flood-environmental-damage", methods=["GET"])
@cross_origin()
def get_flood_assessment():
    data = get_flood_assessment_data()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)


@app.route('/predict-image-landslide-environmental', methods=['POST'])
@cross_origin()
def predict_landslide():
    access = auth_middleware(request)
    if access is not True:
        return access
    
    uploaded_file = request.files['file']
    user_image_path = get_image_path(uploaded_file.filename)
    uploaded_file.save(user_image_path)
    
    date = request.form['date']
    location = request.form['location']
    disaster_type = request.form['disaster_type']
    
    # Load your trained landslide model
    custom_objects = {'dice_coefficient': dice_coefficient}
    landslide_model = tf.keras.models.load_model('landslide_damage_model.h5', custom_objects=custom_objects)
    
    # Preprocess the user-provided image
    user_image = tf.keras.preprocessing.image.load_img(user_image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    input_arr = tf.keras.preprocessing.image.img_to_array(user_image)
    input_arr = input_arr / 255.0
    
    # Predict the mask for the user-provided image
    user_prediction = landslide_model.predict(np.array([input_arr]))
    
    # Define a threshold value
    threshold = 0.35
    
    # Convert the predicted mask to binary
    binary_mask = np.where(user_prediction > threshold, 1, 0)
    
    # Calculate the percentage of landslide damage
    percentage_damage = (np.sum(binary_mask) / np.prod(binary_mask.shape)) * 100
    formatted_percentage = "{:.2f}".format(percentage_damage)
     
    response = {
        "date": date,
        "disaster_type": disaster_type,
        "location": location,
        "percentage_damage": formatted_percentage
    }
    
    add_landslide_assessment_data(response)

    #return jsonify(response), 200
    return str(response)


@app.route("/predict-landslide-environmental", methods=["POST"])
@cross_origin()
def predict_landslide_img():
    access = auth_middleware(request)
    if access is not True:
        return access
    uploaded_file = request.files["file"]
    image_path = get_image_path(uploaded_file.filename)
    uploaded_file.save(image_path)
    prediction = processor.predict_landslide_img(image_path)
    return prediction.title()


@app.route("/assessed-landslide-environmental-damage", methods=["GET"])
@cross_origin()
def get_landslide_assessment():
    data = get_landslide_assessment_data()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)


@app.route('/predict-disaster', methods=['POST'])
@cross_origin()
def predict_disaster_route():
    access = auth_middleware(request)
    if access is not True:
        return access

    date = request.form['date']
    location = request.form['location']

    # Call the predict_disaster function from disaster_forecast_predict.py
    prediction_result = predict_disaster(date, location)

    add_disaster_forecast_data(prediction_result)

    #return jsonify(prediction_result), 200
    return str(prediction_result)


@app.route('/get-disaster-forecasts', methods=['GET'])
@cross_origin()
def get_disaster_forecasts():
    access = auth_middleware(request)
    if access is not True:
        return access

    forecasts = get_disaster_forecast_reports()

    return jsonify(forecasts), 200


@app.route("/knowledge-graph", methods=["GET"])
@cross_origin()
def knowledge_graph():
    access = auth_middleware(request)
    if access is not True:
        return access

    # Fetch data from all relevant collections
    damage_reports_data = get_all_assessment_data()
    flood_damage_data = get_all_assessment_data()
    landslide_damage_data = get_landslide_damage_data()
    forecast_reports = get_disaster_forecast_data()

    # Combine all data
    all_data = damage_reports_data + flood_damage_data + landslide_damage_data

    # Generate the knowledge graph
    graph = generate_knowledge_graph(all_data, forecast_reports)

    # Visualize the graph
    visualize_graph(graph)

    return {"status": "success", "message": "knowledge_graph_generated"}


@app.route('/adaderana_scrape', methods=['POST'])
@cross_origin()
def adaderana_scraping_rout():
    access = auth_middleware(request)
    if access is not True:
        return access
    
    user_query = request.form['user_query']
    # Call the scraping function from humanitarian_risk_assessment.py
    data = fetch_data_from_web(user_query)
    floodlist_data = fetch_floodlist_data_from_web()
    # data_to_insert = data.to_dict('records')
    add_ada_derana_scraped_data(data)
    add_floodlist_scraped_data(floodlist_data)
    #return jsonify(prediction_result), 200
    return str(data)


@app.route('/disaster_report_form', methods=['POST'])
@cross_origin()
def disaster_report():
    access = auth_middleware(request)
    if access is not True:
        return access
    
    # Capture data from the form
    incdnt_date = request.form['incdnt_date']
    report_date_time = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
    location = request.form['location']
    spec_name = request.form['spec_name']
    disaster_type = request.form['disaster_type']
    severity_level = request.form['severity_level']
    death_count = int(request.form['death_count'])
    injured_count = int(request.form['injured_count'])
    missing_count = int(request.form['missing_count'])
    evacuated_count = int(request.form['evacuated_count'])
    medical_availability = request.form['medical_availability']
    hospitalized_count = int(request.form['hospitalized_count'])
    clean_water_availability = request.form['clean_water_availability']
    food_availability = request.form['food_availability']
    shelter_availability = request.form['shelter_availability']
    communication_disruption = request.form['communication_disruption']
    transport_disruption = request.form['transport_disruption']
    additional_comment = request.form['additional_comment']

    # Create a dictionary with the captured data
    report = {
        'incdnt_date' : incdnt_date,
        'report_date_time': report_date_time,
        'location': location,
        'spec_name': spec_name,
        'disaster_type': disaster_type,
        'severity_level': severity_level,
        'death_count': death_count,
        'injured_count': injured_count,
        'missing_count': missing_count,
        'evacuated_count': evacuated_count,
        'medical_availability': medical_availability,
        'hospitalized_count': hospitalized_count,
        'clean_water_availability': clean_water_availability,
        'food_availability': food_availability,
        'shelter_availability': shelter_availability,
        'communication_disruption': communication_disruption,
        'transport_disruption': transport_disruption,
        'additional_comment': additional_comment
    }

    add_disaster_report_forms(report)
    #return jsonify(prediction_result), 200
    return str(report)

@app.route('/get-news', methods=['GET'])
@cross_origin()
def get_disaster_news():
    access = auth_middleware(request)
    if access is not True:
        return access
    disaster_news = get_floodlist_news()
    return jsonify(disaster_news), 200

@app.route('/get-adaderana', methods=['GET'])
@cross_origin()
def get_adaderana():
    access = auth_middleware(request)
    if access is not True:
        return access
    adaderana_news = get_adaderana_news()
    return jsonify(adaderana_news), 200