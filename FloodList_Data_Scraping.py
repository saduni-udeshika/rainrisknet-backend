from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from time import sleep
from random import randint
from db import floodlisd_id_exists_in_mongo
import re

def fetch_floodlist_data_from_web():
  url = "https://floodlist.com/tag/sri-lanka"
  user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
  headers = {'User-Agent': user_agent}
  page = requests.get(url, headers=headers, verify=False)
  soup = BeautifulSoup(page.text, 'html.parser')

  soup.find('main', class_='site-main')

  page_details = soup.find('nav', class_='pagination loop-pagination')
  all_pages = page_details.find_all('a', class_='page-numbers')
  max_pages = [page.text.strip() for page in all_pages]
  pages_count = int(max_pages[1])  # Convert the second element (index 1) to an integer

  my_array = []

  # Generate the array from 1 to n
  for i in range(1, pages_count + 1):
      my_array.append(i)

  for pages in my_array:
    pages = "https://floodlist.com/tag/sri-lanka/page/"+str(pages)
    print(pages)

  post_id = []
  post_title = []
  post_summary = []
  post_url = []
  post_content = []

  for pages in my_array:
      user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
      headers = {'User-Agent': user_agent}
      page = requests.get("https://floodlist.com/tag/sri-lanka/page/"+str(pages), headers=headers, verify=False)
      soup = BeautifulSoup(page.text, 'html.parser')
      main_content = soup.find('main', class_='site-main')
      content = main_content.find_all('article')
      for article in content:
        article_id = article.get('id')
        # Check if the post ID already exists in MongoDB
        if floodlisd_id_exists_in_mongo(article_id):
          continue  # Skip this post
        post_id.append(article_id)
        title = article.h2.a.text.strip()
        post_title.append(title)
        summary_div = article.find('div', class_='entry-summary')
        summary_para = summary_div.find('p')
        summary = summary_para.text.strip()
        post_summary.append(summary)
        url_div = article.find('div', class_='more-link')
        url = url_div.a.get('href')
        post_url.append(url)
        # Extract content from the URL
        user_agent2 = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        headers = {'User-Agent': user_agent2}
        response = requests.get(url, headers=headers, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')
        news_div = soup.find('div', class_='entry-content')
        paragraphs = news_div.find_all('p')
        # Extract text from each <p> tag and store it in a list
        paragraph_texts = [p.get_text() for p in paragraphs]
        # Join the paragraph texts into a single string, if needed
        content_text = '\n'.join(paragraph_texts)
        post_content.append(content_text)

  a = {'Post ID' : post_id, 'Post Title' : post_title, 'Post Summary' : post_summary, 'Post URL' : post_url, 'Post Content' : post_content}
  df = pd.DataFrame.from_dict(a, orient='index')
  df = df.transpose()
  df

  # Function to remove special characters using regular expressions
  def clean_news_content(post_content):
  # Use a regular expression to match and replace non-alphanumeric characters
      cleaned_content = re.sub(r'[^a-zA-Z0-9\s]', '', post_content)
      cleaned_content = cleaned_content.replace('\n', '')
      cleaned_content = ' '.join(cleaned_content.split())
      return cleaned_content

  # Apply the cleaning function to the 'News Title' column
  df['Post Content'] = df['Post Content'].apply(clean_news_content)
  
  return df

# add_floodlist_scraped_data(df)