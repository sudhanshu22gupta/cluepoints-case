import sys
sys.path.append("/Users/sudhanshugupta/Library/Python/3.9/lib/python/site-packages")

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

from transformers import AutoTokenizer
import torch
from typing import List
import re
import pandas as pd
from bs4 import BeautifulSoup

class preprocMovieReview:

    def __init__(self, ds_review_text) -> None:
        """
         Initialize the class.
         
         @param ds_review_text - pd.Series with reviews
        """
        self.ds_review_text = ds_review_text
        self.ds_review_text_filtered = pd.Series(['']*len(ds_review_text))

    def basic_text_sanitization_pipeline(self):
        """
        Basic text sanitization pipeline. Removes special characters tokenizes stopwords stemming and strips whitespace.
        
        
        @return a dataframe with sanitized review text for use in review
        """

        for i, review_text in enumerate(self.ds_review_text):
            review_text =  self.remove_special_characters(review_text)
            tokens = self.tokenize(review_text)
            tokens = self.remove_stopwords(tokens)
            # tokens = self.stemming(tokens)

            review_text_filtered = ' '.join(tokens)
            self.ds_review_text_filtered.iloc[i] = review_text_filtered

        return self.ds_review_text_filtered

    def distilbert_text_sanitization_pipeline(self, max_len):
        """
         This pipeline takes a list of review texts and sanitizes them to a format compatible with (Distil)BERT Classifer
         
         @param max_len - The maximum length of sequence
         
         @return tensors of tokenized sentence and their attention masks
        """
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')
        # token ID storage
        input_ids = []
        # attention mask storage
        attention_masks = []
        # for every review:
        for review_text in self.ds_review_text:
            # This will:
            # Tokenize the sentence.
            # Prepend the `[CLS]` token to the start.
            # Append the `[SEP]` token to the end.
            # Map tokens to their IDs.
            # Pad or truncate the sentence to `max_length`
            # Create attention masks for [PAD] tokens.
            encoded_dict = self.tokenizer.encode_plus(
                                review_text,  # document to encode.
                                add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                                max_length=max_len,  # set max length
                                truncation=True,  # truncate longer messages
                                padding='max_length',  # add padding
                                return_attention_mask=True,  # create attn. masks
                                return_tensors='pt'  # return pytorch tensors
                        )

            # add the tokenized sentence to the list
            input_ids.append(encoded_dict['input_ids'])
            # and its attention mask
            attention_masks.append(encoded_dict['attention_mask'])

        return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

    def remove_special_characters(self, review_text):
        """
         Remove special characters from review text. 
         This is used to prevent HTML tags from appearing in the review text 
         and also remove non alpha numeric characters
         
         @param review_text - Text to be cleaned. Should be plain text.
         
         @return Clean text
        """
        
        # Remove HTML Tags
        html_parser = BeautifulSoup(review_text, "html.parser")
        review_text = html_parser.get_text()

        # Remove non alpha numeric
        pattern = r'[^a-zA-z0-9]'
        review_text = re.sub(pattern, ' ', review_text)
        review_text = review_text.replace("\\", "")

        # Replace multiple spaces with a single space
        review_text = re.sub(r'\s+', ' ', review_text)

        return review_text

    def tokenize(self, review_text):
        """
        Tokenize and strip whitespace. This is used to tokenize a review's text into a list of tokens.
        
        @param review_text - The text to tokenize.
        
        @return A list of tokens in the format described in word_tokenize
        """
        tokens = word_tokenize(review_text)
        tokens = [token.strip() for token in tokens]
        return tokens

    def remove_stopwords(self, tokens: List):
        """
        Remove stopwords (words with low meaning) from a list of tokens.
        
        @param tokens - List of tokens to be filtered.
        
        @return List of filtered tokens after removing stopwords from it
        """

        stopword_list = stopwords.words('english')
        tokens = [token.lower() for token in tokens if token.lower() not in stopword_list]
        return tokens

    def stemming(self, tokens: List):
        """
        Stem a list of tokens. This is a wrapper around PorterStemmer
        
        @param tokens - The list of tokens to stem
        
        @return The list of stemmed tokens
        """

        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        return tokens

class preprocMedicalAppointment:

    def __init__(self) -> None:
        pass

    def remove_outliers_age(self, df_medical_appointment, keep_range=(0, 100)):
        """
         Remove records whose age is outside the keep_range.
         
         @param df_medical_appointment - Data frame of appointments
         @param keep_range - Range to keep ( min max )
         
         @return Data frame of the same type as df_medical
        """
        
        return df_medical_appointment.loc[
            (df_medical_appointment['Age'] >= keep_range[0])
            & (df_medical_appointment['Age'] <= keep_range[1])
            ]