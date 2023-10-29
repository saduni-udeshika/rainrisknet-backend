import numpy as np
import seaborn as sns
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, GlobalMaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import requests

import pandas as pd
import numpy as np
from time import sleep
from random import randint

df = pd.read_csv("disaster_train.csv")
df.head()

df["target"].value_counts()

df = df.drop(['Image Url', 'News Referenace', 'News Title', 'Date and Time', 'Headline'], axis=1)
df = df.drop_duplicates()

df.head()

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

# Check the data type of the "News Content" column
print(df["News Content"].dtype)
# Convert non-string elements to strings
df["News Content"] = df["News Content"].astype(str)
# Now apply the clean_text function
df["News Content"] = df["News Content"].apply(clean_text)

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X.head()
y.head()

X_train,X_test,y_train,y_test = train_test_split(X, y,random_state=0, test_size=0.1)

max_vocab = 20000000
tokenizer = Tokenizer(num_words = max_vocab)
tokenizer.fit_on_texts(X_train.iloc[:, 0])

wordidx = tokenizer.word_index
V = len(wordidx)
print("Dataset vocab size =", V)

train_seq = tokenizer.texts_to_sequences(X_train.iloc[:, 0])
test_seq = tokenizer.texts_to_sequences(X_test.iloc[:, 0])

print("Training Sequence", train_seq[0])
print("Testing Sequence", test_seq[0])

train_pad = pad_sequences(train_seq)
test_pad = pad_sequences(test_seq, maxlen=train_pad.shape[1])
print("Length of training sequence:", train_pad.shape[1])
print("Length of testing sequence:", test_pad.shape[1])

train_pad[0]

T = train_pad.shape[1]
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

model = Model(i, x)

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])
r = model.fit(train_pad,
              y_train,
              validation_data=(test_pad, y_test),
              epochs=6)
r.history

# plt.figure(figsize=(3,3))
# plt.plot(r.history["loss"], label="Loss")
# plt.plot(r.history["val_loss"], label="Validation Loss")
# plt.legend()
# plt.show()
model.save("disaster_model.h5")

def predict_sentiment(text):
  text_seq = tokenizer.texts_to_sequences(text)
  text_pad = pad_sequences(text_seq, maxlen=T)

  prediction = model.predict(text_pad).round()

  if prediction==1.0:
    print("It is a disaster related news.")
  else:
    print("It is not a disaster related news.")
#Prediction Example
# text="What if we used drones to help firefighters lead people out of burning buildings/ help put the fire out?"
text = "Soldiers and other rescuers have discovered remains of 24 people from the Aranayaka landslide site, an official said."
predict_sentiment([text])

#User Inputs - Search Query #Southwest monsoon #aranayaka #flood
user_query = 'aranayaka landslide'
search_query = user_query.replace(" ", "%20")
print(search_query)
# URL for the query
url = "https://www.adaderana.lk/search_results.php?mode=1&show=1&query="+str(search_query)
print(url)

page = requests.get(url, verify=False)
soup = BeautifulSoup(page.text, 'html')
soup.find_all('div', class_='story-text')
print(soup)

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
  newsTitle = url_and_name.a.text.strip()
  NewsTitle.append(newsTitle)
  newsUrl = url_and_name.a.get('href')
  concat_url = "http://adaderana.lk/"+str(newsUrl)
  NewsRef.append(concat_url)
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
  user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
  headers = {'User-Agent': user_agent}
  response = requests.get(concat_url, headers=headers, verify=False)
  soup = BeautifulSoup(response.text, 'html.parser')
  news_div = soup.find('div', class_='news-content')
  paragraphs = news_div.find_all('p')
  # Extract text from each <p> tag and store it in a list
  paragraph_texts = [p.get_text() for p in paragraphs]
  # Join the paragraph texts into a single string, if needed
  content_text = '\n'.join(paragraph_texts)
  NewsContent.append(content_text)

news_contains = {'News Title' : NewsTitle, 'News Referenace' : NewsRef, 'Image Url' : ImgUrl, 'Date and Time' : Date, 'Headline' : HeadLine, 'News Content': NewsContent, 'Disaster Related' : DisasterRelated}
df = pd.DataFrame.from_dict(news_contains, orient='index')
df = df.transpose()
df
from datetime import datetime
import re
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
df['Date and Time'].fillna('Unknown', inplace=True)
df
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
df