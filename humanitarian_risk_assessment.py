from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import re
from db import post_id_exists_in_mongo

#New Imports
import seaborn as sns
import tensorflow as tf
# import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, GlobalMaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.models import load_model
from time import sleep
from random import randint
##################
#Newly Added
# Load the saved model
model = load_model("disaster_model.h5")

# Define a function to clean the text (same function as in the first code)
def clean_text(text):
    text = re.sub(r'#', '', text)
    text = re.sub(r'@[-)]+', '', text)
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'@[A-Za-z]+', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?\/\/\S+', '', text)
    text = re.sub(r'&[a-z;]+', '', text)
    text = re.sub(r'[WATCH]', '', text)
    return text
######################

def fetch_data_from_web(user_query):
    search_query = user_query.replace(" ", "%20")
# URL for the query
    url = "https://www.adaderana.lk/search_results.php?mode=1&show=1&query="+str(search_query)

    page = requests.get(url, verify=False)
    soup = BeautifulSoup(page.text, 'html.parser')
    soup.find_all('div', class_='story-text')
    content = soup.find_all('div', class_='story-text')

    NewsTitle = []
    NewsRef = []
    ImgUrl = []
    Date = []
    HeadLine = []
    NewsContent = []
    DisasterRelated = []

    for div in content:
            url_and_name = div.find('h4', class_ = 'hidden-xs')
            newsUrl = url_and_name.a.get('href')
            concat_url = "http://adaderana.lk/"+str(newsUrl)
            if post_id_exists_in_mongo(concat_url):
                continue  # Skip this post
            NewsRef.append(concat_url)
            newsTitle = url_and_name.a.text.strip()
            NewsTitle.append(newsTitle)
            image_div = div.find('div', class_ = 'thumb-image')
            image = image_div.img.get('src')
            ImgUrl.append(image)
            date_div = div.find('div', class_ = 'comments pull-right hidden-xs')
            date = date_div.a.text.strip()
            Date.append(date)
            head_para = div.find('p', class_ = 'news')
            head = head_para.text.strip()
            head_split = head.split('MORE..')[0]
            HeadLine.append(head_split)
            # Extract content from the URL
            news_content = extract_content_from_url(concat_url)
            NewsContent.append(news_content)

    news_contains = {'News Title' : NewsTitle, 'News Referenace' : list(NewsRef), 'Image Url' : ImgUrl, 'Date and Time' : Date, 'Headline' : HeadLine, 'News Content': NewsContent}
    df = pd.DataFrame.from_dict(news_contains, orient='index')
    df = df.transpose()

    # Define a list of representations of missing values
    missing_values = ['', 'Unknown', 'NaN', None]

    # Function to clean and format datetime
    def clean_and_format_datetime(date_str):
        try:
            # Parse the date string into a datetime object
            datetime_obj = datetime.strptime(date_str, '%B %d, %Y %I:%M %p')
            # Format it into the desired output format (e.g., 'YYYY-MM-DD HH:MM:SS')
            return datetime_obj.strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            # Handle invalid date formats or missing values
            return 'Invalid Date'

    df['Date and Time'] = df['Date and Time'].apply(lambda x: clean_and_format_datetime(x) if x not in missing_values else 'Unknown')

    # Function to remove special characters using regular expressions
    def clean_news_title(title):
    # Use a regular expression to match and replace non-alphanumeric characters
        cleaned_title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
        return cleaned_title

    # Apply the cleaning function to the 'News Title' column
    df['News Title'] = df['News Title'].apply(clean_news_title)

    # Function to remove special characters using regular expressions
    def clean_news_content(content):
    # Use a regular expression to match and replace non-alphanumeric characters
        cleaned_content = re.sub(r'[^a-zA-Z0-9\s]', '', content)
        # Replace multiple spaces with a single space
        cleaned_content = ' '.join(cleaned_content.split())
        return cleaned_content

    # Apply the cleaning function to the 'News Content' column
    df['News Content'] = df['News Content'].apply(clean_news_content)

    # Check the data type of the "News Content" column
    print(df["News Content"].dtype)
    # Convert non-string elements to strings
    df["News Content"] = df["News Content"].astype(str)
    # Now apply the clean_text function
    df["News Content"] = df["News Content"].apply(clean_text)

    print()

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X.head()
    y.head()

    max_vocab = 20000000
    tokenizer = Tokenizer(num_words = max_vocab)

    wordidx = tokenizer.word_index
    V = len(wordidx)
    print("Dataset vocab size =", V)

    T = 418
    D = 20
    M = 15

    i = Input(shape=(T, ))
    x = Embedding(V+1, D)(i)
    x = LSTM(M, return_sequences=True)(x)
    x = Dropout(0.1)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.6)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.7)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.7)(x)
    x = Dense(1, activation="sigmoid")(x)

    # Create a function to preprocess and predict the sentiment for a single news article
    def process_and_predict_sentiment(headline):
        # Clean and preprocess the text
        cleaned_text = clean_text(headline)
        # Tokenize and pad the text
        text_seq = tokenizer.texts_to_sequences([cleaned_text])
        text_pad = pad_sequences(text_seq, maxlen=T)
        # Predict the sentiment
        prediction = model.predict(text_pad).round().astype(int)
        
        if prediction == 1:
            DisasterRelated.append("yes")
            return "Disaster-related news"
        else:
            return "Non-disaster-related news"

    # Iterate through the Headlines of web-scraped news articles
    for headline in df['News Content']:
        prediction = process_and_predict_sentiment(headline)
        print(f"News: {headline}\nPrediction: {prediction}\n")

    # Add the 'DisasterRelated' column to the DataFrame
    df['DisasterRelated'] = DisasterRelated
    print(df)
    return df

# def save_data_to_mongodb(data):
#     collection = mongo.db['AdaDeranaNews']  # Replace with your collection name
#     data_to_insert = data.to_dict('records')
#     collection.insert_many(data_to_insert)

def extract_content_from_url(url):
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    headers = {'User-Agent': user_agent}
    response = requests.get(url, headers=headers, verify=False)
    soup = BeautifulSoup(response.text, 'html.parser')
    news_div = soup.find('div', class_='news-content')
    paragraphs = news_div.find_all('p')
    # Extract text from each <p> tag and store it in a list
    paragraph_texts = [p.get_text() for p in paragraphs]
    # Join the paragraph texts into a single string, if needed
    content_text = '\n'.join(paragraph_texts)
    
    return content_text

# def save_data_to_mongodb(data):
#     collection = mongo.db['AdaDeranaNews']  # Replace with your collection name
#     data_to_insert = data.to_dict('records')
#     if data is not None and not data.empty:
#         collection.insert_many(data_to_insert)

if __name__ == '__main__':
    # app.run(debug=True)
    input_query = input("Enter the Search Query: ")
        