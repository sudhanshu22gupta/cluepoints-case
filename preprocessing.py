import sys
sys.path.append("/Users/sudhanshugupta/Library/Python/3.9/lib/python/site-packages")

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

from transformers import AutoTokenizer

from typing import List
import re
import pandas as pd
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

class preprocMovieReview:

    def __init__(self, ds_review_text) -> None:
        self.ds_review_text = ds_review_text
        self.ds_review_text_filtered = pd.Series(['']*len(ds_review_text))

    def basic_text_sanitization_pipeline(self):

        for i, review_text in enumerate(self.ds_review_text):
            review_text =  self.remove_special_characters(review_text)
            tokens = self.tokenize(review_text)
            tokens = self.remove_stopwords(tokens)
            tokens = self.stemming(tokens)

            review_text_filtered = ' '.join(tokens)
            self.ds_review_text_filtered.iloc[i] = review_text_filtered

        return self.ds_review_text_filtered

    def bert_text_sanitization_pipeline(self):
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        for i, review_text in enumerate(self.ds_review_text):
            review_text_filtered = tokenizer(review_text, truncation=True)
            self.ds_review_text_filtered.iloc[i] = review_text_filtered
        return self.ds_review_text_filtered

    def remove_special_characters(self, review_text):
        
        # Remove HTML Tags
        html_parser = BeautifulSoup(review_text, "html.parser")
        review_text = html_parser.get_text()

        # Remove non alpha numeric
        pattern = r'[^a-zA-z0-9\s]'
        review_text = re.sub(pattern, ' ', review_text)

        # Replace multiple spaces with a single space
        review_text = re.sub(r'\s+', ' ', review_text)

        return review_text

    def tokenize(self, review_text):

        tokens = word_tokenize(review_text)
        tokens = [token.strip() for token in tokens]
        return tokens

    def remove_stopwords(self, tokens: List):

        stopword_list = stopwords.words('english')
        tokens = [token for token in tokens if token.lower() not in stopword_list]
        return tokens

    def stemming(self, tokens: List):

        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        return tokens

def train_val_test_split_movie_reviews(df, reviews_col, target_col, test_percent: float, val_percent: float=0.0):

    assert test_percent+val_percent < 100
    train_percent = 100 - (test_percent+val_percent)
    total_size = len(df)

    # shuffle the dataset
    df_in = df.copy().sample(total_size)
    
    df_in_train = df_in.iloc[:int((train_percent*total_size)/100)]
    X_train, y_train = df_in_train[reviews_col], df_in_train[target_col].values
    df_in = df_in.iloc[int((train_percent*total_size)/100):]

    df_in_test = df_in.iloc[:int((test_percent*total_size)/100)]
    X_test, y_test = df_in_test[reviews_col], df_in_test[target_col].values
    df_in = df_in.iloc[int((test_percent*total_size)/100):]

    if val_percent:
        df_in_val = df_in
        X_val, y_val = df_in_val[reviews_col], df_in_val[target_col].values
    else:
        X_val, y_val = pd.Series(), pd.Series()
    
    return X_train, y_train, X_test, y_test, X_val, y_val

class Vectorizer:

    def __init__(self, vectorizer_type) -> None:
        
        if vectorizer_type == 'tfidf':
            self.initialize_tfidf_vectorizer()
        elif vectorizer_type == 'count':
            self.initialize_count_vectorizer()
        else:
            raise NotImplementedError
    
    def initialize_tfidf_vectorizer(self):
        self.vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, use_idf=True, ngram_range=(1,3))

    def initialize_count_vectorizer(self):
        self.vectorizer = CountVectorizer(min_df=0.0, max_df=1.0, binary=False, ngram_range=(1,3))

    def apply_transform_train(self, reviews):
        return self.vectorizer.fit_transform(reviews)

    def apply_transform_test(self, reviews):
        return self.vectorizer.transform(reviews)