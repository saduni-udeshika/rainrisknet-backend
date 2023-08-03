import pymongo

# TODO: change mongo url
client = pymongo.MongoClient(
    "mongodb+srv://server:Sigen123@sigen.zkpx7.mongodb.net/sigen?retryWrites=true&w=majority")
db = client['sigen']
collection = db['net_users']


def add_user(user_data):
    result = collection.insert_one(user_data)
    return result.inserted_id


def get_user(email):
    result = collection.find_one({"email": email})
    return result
