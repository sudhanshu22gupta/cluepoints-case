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

    def distilbert_text_sanitization_pipeline(self, max_len):
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')
        # token ID storage
        input_ids = []
        # attention mask storage
        attention_masks = []
        # for every review:
        for review_text in self.ds_review_text:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
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
            # and its attention mask (differentiates padding from non-padding)
            attention_masks.append(encoded_dict['attention_mask'])

        return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)


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