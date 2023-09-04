import pymongo

client = pymongo.MongoClient(
    "mongodb+srv://admin123:admin123@rainrisknet.nbql448.mongodb.net/?retryWrites=true&w=majority"
)
db = client["damage_assessment"]
user_collection = db["user"]
damage_reports_collection = db["damage_reports"]
flood_damage_reports_collection = db["flood_damage_reports"]
landslide_damage_reports_collection = db["landslide_damage_reports"]
disaster_forecast_reports_collection = db["disaster_forecast_reports"]

def add_user(user_data):
    result = user_collection.insert_one(user_data)
    return result.inserted_id


def get_user(email):
    result = user_collection.find_one({"email": email})
    return result


def add_assessment_data(payload):
    result = damage_reports_collection.insert_one(payload)
    return result


def get_assessment_data(email):
    result = damage_reports_collection.find({"email": email})
    data = []
    for x in result:
        data.append(
            {
                "_id": str(x["_id"]),
                "disaster_type": x["disaster_type"],
                "location": x["location"],
                "date": x["date"],
                "damage_percentage": x["damage_percentage"],
            }
        )
    return data



def get_all_assessment_data():
    result = damage_reports_collection.find()
    data = []
    for x in result:
        item = {
            "_id": str(x["_id"]),
            "disaster_type": x.get("disaster_type", ""),
            "location": x.get("location", ""),
            "date": x.get("date", ""),
            "damage_percentage": x.get("damage_percentage", ""),
        }
        data.append(item)

    result = flood_damage_reports_collection.find()
    for x in result:
        item = {
            "_id": str(x["_id"]),
            "disaster_type": x.get("disaster_type", ""),
            "location": x.get("location", ""),
            "date": x.get("date", ""),
            "percentage_damage": x.get("percentage_damage", ""),
        }
        data.append(item)
        
    return data


def get_landslide_damage_data():
    result = landslide_damage_reports_collection.find()
    data = []
    for x in result:
        item = {
            "_id": str(x["_id"]),
            "disaster_type": x.get("disaster_type", ""),
            "location": x.get("location", ""),
            "date": x.get("date", ""),
            "percentage_damage": x.get("percentage_damage", ""),
        }
        data.append(item)
    return data


def get_disaster_forecast_data():
    reports = list(disaster_forecast_reports_collection.find({}, {'_id': 0}))
    return reports



def add_flood_assessment_data(payload):
    result = flood_damage_reports_collection.insert_one(payload)
    return result


def get_flood_assessment_data():
    result = flood_damage_reports_collection.find()
    data = []
    for x in result:
        item = {
            "_id": str(x["_id"]),
            "disaster_type": x.get("disaster_type", ""),
            "location": x.get("location", ""),
            "date": x.get("date", ""),
            "percentage_damage": x.get("percentage_damage", ""),
        }
        data.append(item)
    return data


def add_landslide_assessment_data(payload):
    result = landslide_damage_reports_collection.insert_one(payload)
    return result


def get_landslide_assessment_data():
    result = landslide_damage_reports_collection.find()
    data = []
    for x in result:
        item = {
            "_id": str(x["_id"]),
            "disaster_type": x.get("disaster_type", ""),
            "location": x.get("location", ""),
            "date": x.get("date", ""),
            "percentage_damage": x.get("percentage_damage", ""),
        }
        data.append(item)
    return data    


def add_disaster_forecast_data(payload):
    result = disaster_forecast_reports_collection.insert_one(payload)
    return result

def get_disaster_forecast_reports():
    reports = list(disaster_forecast_reports_collection.find({}, {'_id': 0}))
    return reports
