#!/usr/bin/env python
# coding: utf-8

# In[209]:


import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split


# In[210]:

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz
#          https://web.archive.org/web/20201127142707if_/https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Office_Products_v1_00.tsv.gz


# # Dataset Preparation
# 

# ## Read Data
# 

# In[211]:


filepath = "data.tsv"

df = pd.read_csv(filepath, sep='\t', engine="python", on_bad_lines="skip")

print(df.head())
print(df.shape)


# ## Keep Reviews and Ratings
# 

# In[212]:


cols = ["review_body", "star_rating"]
df = df[cols]


# In[213]:


with pd.option_context('display.max_colwidth', None):
    print(df.iloc[0])


# In[214]:


with pd.option_context('display.max_colwidth', None):
    print(df.iloc[4])


# In[215]:


with pd.option_context('display.max_colwidth', None):
    print(df.iloc[6])


# In[216]:


# 1a) rating statistics
rating_counts = df['star_rating'].value_counts()
print(rating_counts)


# ## Relabeling and Sampling
# 
# First form three classes and print their statistics. Then randomly select 100,000 reviews from the positive and 100,000 reviews from the negative
# 

# In[217]:


# create binary labels
df["sentiment"] = -1
df.loc[df["star_rating"] >= 4, "sentiment"] = 1
df.loc[df["star_rating"] <= 2, "sentiment"] = 0

# count sentiment classes
sentiment_counts = df["sentiment"].value_counts()
print(sentiment_counts)


# In[218]:


# discard rating 3 reviews (-1 sentiment)
df = df[df["sentiment"] != -1]
sentiment_counts = df["sentiment"].value_counts()
print(sentiment_counts)
print(df.columns)


# In[219]:


# randomly select 100,000 positive and 100,000 negative sentiment reviews
n = 100000

df = (
    df.groupby("sentiment", group_keys=False)
    .sample(n=n, random_state=42)
)

print(df["sentiment"].value_counts())


# # Data Cleaning
# 

# In[220]:


# print average length of reviews by char length BEFORE cleaning
avg_len = df["review_body"].str.len().mean()
print("Avg len before cleaning:", round(avg_len))

# print 3 sample reviews BEFORE cleaning and preprocessing
with pd.option_context('display.max_colwidth', None):
    print(df["review_body"][:3])

# convert all reviews to lowercase
df["review_body"] = df["review_body"].str.lower()

def strip_html_and_urls(text):
    if pd.isna(text):
        return text
    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    return text

# remove extra spaces
df["review_body"] = df["review_body"].str.replace(r"\s+", " ", regex=True)

# perform contractions
contractions = {
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "cannot",
    "couldn't": "could not",
    "i'm": "i am",
    "ain't": "am not",
    "it's": "it is",
    "you're": "you are",
    "they're": "they are",
    "i've": "i have",
    "isn't": "is not",
    "aren't": "are not",
    "don't": "do not",
    "didn't": "did not",
    "doesn't": "does not",
    "musn't": "must not",
    "hasn't": "has not",
    "hadn't": "had not",
    "haven't": "have not"
}

# add the lazy version of contractions (without apostrophe)
# to dict
for k in list(contractions.keys()):
    contractions[k.replace("'", "")] = contractions[k]

def expand_contractions(text):
    if pd.isna(text):
        return text
    pattern = re.compile(r'\b(' + '|'.join(contractions.keys()) + r')\b')
    return pattern.sub(lambda x: contractions[x.group()], text)

df["review_body"] = df["review_body"].apply(expand_contractions)

# remove non-alphabetical characters
df["review_body"] = df["review_body"].str.replace(r"[^a-zA-Z\s]", "", regex=True)

# print average length of reviews by char length AFTER cleaning
avg_len = df["review_body"].str.len().mean()
print("Avg len after cleaning:", round(avg_len))


# # Pre-processing
# 

# ## remove the stop words
# 

# In[221]:


from nltk.corpus import stopwords
nltk.download("stopwords")

# print average length of reviews by char length BEFORE preprocessing
avg_len = df["review_body"].str.len().mean()
print("Avg len before preprocessing:", round(avg_len))

stop_words = set(stopwords.words("english"))
words_to_keep = {
    "not", "no",
    "will", "would", "can", "could", "am", "is", "are",
    "have", "has", "had", "must", "do", "does", "did"
}
stop_words -= words_to_keep
df["review_body"] = df["review_body"].apply(
    lambda x: " ".join([word for word in x.split() if word not in stop_words])
    if isinstance(x, str) else ""
)


# ## perform lemmatization
# 

# In[222]:


from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()
df["review_body"] = df["review_body"].apply(lambda x: " ".join([lemmatizer.lemmatize(w, pos="v") for w in x.split()]))

# print 3 sample reviews AFTER cleaning and preprocessing
with pd.option_context('display.max_colwidth', None):
    print(df["review_body"][:3], '\n')

# print average length of reviews by char length AFTER preprocessing
avg_len = df["review_body"].str.len().mean()
print("Avg len after preprocessing:", round(avg_len))


# # Bigram Feature Extraction
# 

# In[223]:


import nltk
nltk.download("punkt_tab")
from nltk.util import bigrams
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction import DictVectorizer

def extract_bigrams(tokens):
    bg = bigrams(tokens)
    return Counter(["_".join(pair) for pair in bg])

df["tokens"] = df["review_body"].apply(word_tokenize)

# Convert Series of Counters → list of dicts
bigram_dicts = list(df["tokens"].apply(extract_bigrams))

vec = DictVectorizer(sparse=True)
X_sparse = vec.fit_transform(bigram_dicts)
y = df["sentiment"]


# In[224]:


# convert to dataframe
# split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_sparse,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


"""
total dataset size = 200,000
80% training = 160,000
20% test = 40,000
""" 
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# # Perceptron
# 

# In[225]:


from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

perceptron = Perceptron(
    max_iter=1000,
    tol=1e-3,
    random_state=42
)

perceptron.fit(X_train, y_train)


# In[226]:


def predict_and_print_metrics(model, model_name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # train metrics
    train_accuracy  = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall    = recall_score(y_train, y_train_pred)
    train_f1        = f1_score(y_train, y_train_pred)

    # test metrics
    test_accuracy  = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall    = recall_score(y_test, y_test_pred)
    test_f1        = f1_score(y_test, y_test_pred)

    print(model_name + " Training Accuracy: {:.4f}".format(train_accuracy))
    print(model_name + " Training Precision: {:.4f}".format(train_precision))
    print(model_name + " Training Recall: {:.4f}".format(train_recall))
    print(model_name + " Training F1-score: {:.4f}".format(train_f1))

    print(model_name + " Testing Accuracy: {:.4f}".format(test_accuracy))
    print(model_name + " Testing Precision: {:.4f}".format(test_precision))
    print(model_name + " Testing Recall: {:.4f}".format(test_recall))
    print(model_name + " Testing F1-score: {:.4f}".format(test_f1))

predict_and_print_metrics(perceptron, "Perceptron")


# # SVM
# 

# In[227]:


from sklearn.svm import LinearSVC

svm = LinearSVC(
    random_state=42,
    max_iter=10000
)

svm.fit(X_train, y_train)


# In[228]:


predict_and_print_metrics(svm, "SVM")


# 

# # Logistic Regression
# 

# In[229]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(
    random_state=42,
    max_iter=1000, 
    solver='liblinear'
)

logreg.fit(X_train, y_train)


# In[230]:


predict_and_print_metrics(logreg, "Logistic Regression")


# # Naive Bayes
# 

# In[231]:


from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train, y_train)


# In[232]:


predict_and_print_metrics(nb, "Naive Bayes")

