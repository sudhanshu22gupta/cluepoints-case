from wordcloud import WordCloud, STOPWORDS
from nltk.util import ngrams
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

class EDA_movie_reviews:

    def __init__(self, df_movie_reviews, reviews_col, target_col) -> None:
        self.df_movie_reviews = df_movie_reviews
        self.reviews_col = reviews_col
        self.target_col = target_col
        self.df_movie_reviews["n_tokens"] = self.df_movie_reviews[self.reviews_col].apply(lambda x: len(x.split(' ')))

        self.df_positive_reviews = self.df_movie_reviews.loc[self.df_movie_reviews[self.target_col] == 1]
        self.df_negative_reviews = self.df_movie_reviews.loc[self.df_movie_reviews[self.target_col] == 0]

    def visualize_wordcloud(self):
        fig, axs = plt.subplots(ncols=2, figsize=(18, 10))
        for i, (reviews_type, df_reviews) in enumerate([("POSITIVE", self.df_positive_reviews), ("NEGATIVE", self.df_negative_reviews)]):
            wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords = STOPWORDS).generate(str(df_reviews[self.reviews_col]))
            ax = axs[i]
            ax.set_title(f"{reviews_type} Reviews - Wordcloud")
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
        plt.show()

    def visulaize_class_distribution(self):
        self.df_movie_reviews[self.target_col].value_counts().plot(kind="bar")
        plt.title("Class Distribution\n(0=NEGATIVE, 1=POSITIVE)")
        plt.ylabel("Count")
        plt.show()

    def statistics_on_review_text(self):
        """count of tokens per review"""
        for reviews_type, df_reviews in [("POSITIVE", self.df_positive_reviews), ("NEGATIVE", self.df_negative_reviews)]:
            print(f"Statistics on count of tokens per {reviews_type} movie review")
            display(df_reviews["n_tokens"].describe())

    def show_common_n_grams(self, n=3, show_count=20):
        for reviews_type, df_reviews in [("POSITIVE", self.df_positive_reviews), ("NEGATIVE", self.df_negative_reviews)]:
            text_ngrams = [ngram for review_text in df_reviews[self.reviews_col] for ngram in ngrams(review_text.split(), n)]
            most_common_ngrams = Counter(text_ngrams).most_common(show_count)
            print(f"most common {n}-grams for {reviews_type}:\n{most_common_ngrams}")

class EDA_medical_appointment:

    def __init__(self, df_medical_appointment_features, target_labels) -> None:
        self.df_medical_appointment = df_medical_appointment_features
        self.target_labels = target_labels
        self.numeric_feature_columns = ['Age']
        self.categorical_feature_columns = ['Gender', 'Neighbourhood', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received']

    def describe_general_stats(self):
        display(self.df_medical_appointment.info())
        display(self.df_medical_appointment[self.numeric_feature_columns].describe())
        # self.df_medical_appointment[self.numeric_feature_columns].plot(kind='hist')
        for column in self.categorical_feature_columns:
            display(self.df_medical_appointment[column].value_counts())

    def stats_per_unique_patient(self):
        df_medical_appointment_groupby_patient = self.df_medical_appointment.sort_values(by='AppointmentDay').groupby('PatientID')
        display(df_medical_appointment_groupby_patient.count().sort_values(by=['AppointmentID']))
        n_patients_less_than_3_visits = len(df_medical_appointment_groupby_patient.count().loc[df_medical_appointment_groupby_patient.count()['AppointmentID']<3])
        print(f"n_patients_less_than_3_visits = {n_patients_less_than_3_visits}")
        print(f"percent_patients_less_than_3_visits = {round(100*n_patients_less_than_3_visits/len(df_medical_appointment_groupby_patient.count()))}")
        df_patient_last_appointment = df_medical_appointment_groupby_patient.last()
        for column in self.categorical_feature_columns:
            display(df_patient_last_appointment[column].value_counts())

    def visulaize_class_distribution(self):
        pd.Series(self.target_labels).value_counts().plot(kind="bar")
        plt.title("Class Distribution\n(0=Show, 1=No Show)")
        plt.ylabel("Count")
        plt.show()

    def visulaize_no_show_prob_per_variable(self):

        vars_categorical = self.categorical_feature_columns[:]
        vars_categorical.remove("Neighbourhood")
        fig, axs = plt.subplots(figsize=(len(vars_categorical)*2, 5), ncols=len(vars_categorical), sharey=True)
        for cat_var, ax in zip(vars_categorical, axs):
            cat_var_counts = self.df_medical_appointment[cat_var].value_counts() / len(self.df_medical_appointment)
            cat_var_counts = cat_var_counts.loc[~(cat_var_counts.index==0)]
            cat_var_counts.plot(kind="bar", ax=ax)
            ax.set_xlabel(cat_var)
            ax.set_ylabel("probability of No Show")
            ax.set_ylim(0, 1)
        plt.suptitle(f"feature dependency on no-show status")
        plt.tight_layout()
        plt.show()

        