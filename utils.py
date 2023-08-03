import os
import jwt
import bcrypt

CURRENT_DIR = os.getcwd()
SECRET = "J5XC1G1685CA2DXMKOG"


def get_image_path(file_name):
    return os.path.join(CURRENT_DIR, 'uploads', file_name)


def generate_access_token(email):
    return jwt.encode({"email": email}, SECRET, algorithm="HS256")


def decode_access_token(token):
    return jwt.decode(token, SECRET, algorithms="HS256")


def validate_access(token):
    try:
        decode_access_token(token)
        return True
    except:
        return False


def encrypt_password(password):
    encoded_password = password.encode("utf-8")
    return bcrypt.hashpw(encoded_password, bcrypt.gensalt())


def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password)
