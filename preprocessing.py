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
    

class preprocMedicalAppointment:

    def __init__(self) -> None:
        pass

    def remove_outliers_age(self, df_medical_appointment, keep_range=(0, 100)):
        return df_medical_appointment.loc[
            (df_medical_appointment['Age'] >= keep_range[0])
            & (df_medical_appointment['Age'] <= keep_range[1])
            ]