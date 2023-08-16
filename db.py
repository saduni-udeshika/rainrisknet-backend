import pymongo

# TODO: change mongo url
client = pymongo.MongoClient(
    "mongodb+srv://server:Sigen123@sigen.zkpx7.mongodb.net/sigen?retryWrites=true&w=majority"
)
db = client["damage_assessment"]
user_collection = db["user"]
damage_reports_collection = db["damage_reports"]


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
