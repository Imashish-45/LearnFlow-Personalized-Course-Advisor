#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Imashish-45/LearnFlow-Personalized-Course-Advisor/blob/main/Course_recommender.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 
# 
# ## Introduction to Content-Based Recommendation System
# 
# Welcome to this notebook exploring content-based recommendation systems! In the world of personalized recommendations, content-based systems play a significant role by suggesting items to users based on the inherent characteristics of those items. These systems leverage the idea that users who liked a particular item in the past are likely to be interested in items with similar attributes or content.
# 
# The foundation of a content-based recommendation system lies in the understanding of item attributes, features, or descriptions. By analyzing and quantifying these characteristics, the system can establish connections between items and tailor recommendations to match a user's preferences.
# 
# **Key Features of Content-Based Recommendation Systems:**
# 
# - **Personalization**: Content-based systems provide personalized recommendations by focusing on the attributes that matter most to the user. They're capable of suggesting items that align closely with the user's tastes.
# 
# - **Cold Start Problem**: One of the strengths of content-based systems is their ability to handle the "cold start" problem. Even when a user is new or an item is just introduced, the system can make recommendations based on the item's attributes.
# 
# - **Item Diversity**: While content-based systems offer recommendations tailored to a user's preferences, they can sometimes fall short in terms of diversity, as recommendations are driven by item attributes.
# 
# In this notebook, we'll explore the concepts behind content-based recommendation systems and walk through the process of building one step by step. By the end, you'll have a deeper understanding of how these systems work and how you can apply them to provide personalized recommendations to users.
# 
# So, let's dive in and uncover the inner workings of content-based recommendation systems!
# 
# ---
# 
# 

# In[ ]:


# importing important libraries:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, KFold


# ## Data loading and exploration:

# In[ ]:


# df = pd.read_csv("/content/udemy_output_All_IT__Software_p1_p626.csv")
df = pd.read_csv(r"C:\Users\aupadhyay\Documents\devo\udemy_output_All_IT__Software_p1_p626.csv")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.shape


# # **Data Cleaning:**

# In[ ]:


# Drop the specified columns
columns_to_drop = ['discount_price__currency', 'price_detail__currency']
df = df.drop(columns=columns_to_drop)


# In[ ]:


# Removing the currency symbols:

columns_to_clean = ['price_detail__price_string', 'discount_price__price_string']
for column in columns_to_clean:
    df[column].replace('₹', '', regex=True, inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.info()


# ## Text Preprocessing:

# In[ ]:


# Downloading NLTK fucntionalities
nltk.download('stopwords')
nltk.download('punkt')


# In[ ]:


def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space,
    # which in effect deletes the punctuation marks
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)


# In[ ]:


df['title'] = df['title'].apply(remove_punctuation)
df.head(10)


# In[ ]:


# extracting the stopwords from nltk library
sw = stopwords.words('english')
# displaying the stopwords
np.array(sw)


# In[ ]:


print("Number of stopwords: ", len(sw))


# In[ ]:


def stopwords(text):
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(text)


# In[ ]:


df['title'] = df['title'].apply(stopwords)
df.head(10)


# ## Tokenizing:

# In[ ]:


count_vectorizer = CountVectorizer()
# fit the count vectorizer using the text data
count_vectorizer.fit(df['title'])
# collect the vocabulary items used in the vectorizer
dictionary = count_vectorizer.vocabulary_.items()


# In[ ]:


# lists to store the vocab and counts
vocab = []
count = []
# iterate through each vocab and count append the value to designated lists
for key, value in dictionary:
    vocab.append(key)
    count.append(value)
# store the count in panadas dataframe with vocab as index
vocab_bef_stem = pd.Series(count, index=vocab)
# sort the dataframe
vocab_bef_stem = vocab_bef_stem.sort_values(ascending=False)


# # **Vectorization:**

# In[ ]:


# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")  # You can adjust the number of features

# Transform the preprocessed titles into TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(df['title'])

# Print the shape of the TF-IDF matrix
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)


# In[ ]:


tfidf_matrix


# ## **Checking similarities between courses:**

# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity


# # **Visuword :**
# 
# It is a dictionary that allows you to look up words and learn about their origins and similarities with other terms and words. It produces nodes with all of the related terms, as well as the meaning and every aspect of the phrase. The user can tap a node to see a definition for that word category, and press and drag individual nodes to help explain connections.
# 
# "enables users to look up words to find their definitions and connections with other terms and concepts."

# In[ ]:


# Calculating similarity


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# # **Recommendation System:**

# # **The Rising Star algorithm:**
# 
# 
# 
# The Rising Star algorithm generates interpretable feature significance rankings, allowing users to choose the most essential features in the categorization process. The rising star algorithm pruned data to be transferred and on basis of this algorithm, it would do a ranking of course material based on content. With the aid of the rising star algorithm, it automatically estimates the number of important characteristics and the model's complexity, making it suitable for a wide range of datasets. The Rising Star technique is scalable and computationally efficient, allowing it to handle enormous datasets using few processing resources. The top document features have been selected for recommendation to students.

# In[ ]:


def recommend_courses_by_skill(skill, num_recommendations=5):
    # Transform the skill into a TF-IDF vector using the same vectorizer
    skill_tfidf = tfidf_vectorizer.transform([skill])

    # Calculate cosine similarity between the skill and all courses
    similarity_scores = cosine_similarity(skill_tfidf, tfidf_matrix)[0]

    # Get indices of recommended courses
    recommended_indices = similarity_scores.argsort()[-num_recommendations:][::-1]

    # Get recommended course titles
    recommended_courses = df.iloc[recommended_indices]['title'].tolist()

    return recommended_courses


# ## **Testing:**

# In[ ]:


# # Example usage
# user_skill = "API"  #skill user want to learn
# recommended_courses = recommend_courses_by_skill(user_skill)

# if recommended_courses:
#     print(f"Recommended courses for skill '{user_skill}':")
#     for course in recommended_courses:
#         print(course)
# else:
#     print("No recommendations available for the entered skill.")


# In[ ]:


# Hence user can add the skill they want to learn and our model will give 5 suggestion based on user input.

