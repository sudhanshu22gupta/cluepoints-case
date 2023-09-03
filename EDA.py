from wordcloud import WordCloud, STOPWORDS
from nltk.util import ngrams
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class EDA_movie_reviews:

    def __init__(self, df_movie_reviews, reviews_col, target_col) -> None:
        """
         Initialize the class.
         
         @param df_movie_reviews - pandas DataFrame with the reviews for the movie.
         @param reviews_col - string column name of the column that contains the list of ratings.
         @param target_col - string column name of the target column.
        """
        self.df_movie_reviews = df_movie_reviews
        self.reviews_col = reviews_col
        self.target_col = target_col
        self.df_movie_reviews["n_tokens"] = self.df_movie_reviews[self.reviews_col].apply(lambda x: len(x.split(' ')))

        self.df_positive_reviews = self.df_movie_reviews.loc[self.df_movie_reviews[self.target_col] == 1]
        self.df_negative_reviews = self.df_movie_reviews.loc[self.df_movie_reviews[self.target_col] == 0]

    def visualize_wordcloud(self):
        """
         Visualize wordcloud for each review type.
        """
        fig, axs = plt.subplots(ncols=2, figsize=(18, 10))
        # Generate a Wordcloud plot for each review type.
        for i, (reviews_type, df_reviews) in enumerate([("POSITIVE", self.df_positive_reviews), ("NEGATIVE", self.df_negative_reviews)]):
            wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords = STOPWORDS).generate(str(df_reviews[self.reviews_col]))
            ax = axs[i]
            ax.set_title(f"{reviews_type} Reviews - Wordcloud")
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
        plt.show()

    def visulaize_class_distribution(self):
        """
         Visualize the class distribution of the reviews.
        """
        self.df_movie_reviews[self.target_col].value_counts().plot(kind="bar")
        plt.title("Class Distribution\n(0=NEGATIVE, 1=POSITIVE)")
        plt.ylabel("Count")
        plt.show()

    def statistics_on_review_text(self):
        """
         Print statistics on number of tokens per review.
        """
        # Statistics on count of tokens per movie review
        for reviews_type, df_reviews in [("POSITIVE", self.df_positive_reviews), ("NEGATIVE", self.df_negative_reviews)]:
            print(f"Statistics on count of tokens per {reviews_type} movie review")
            display(df_reviews["n_tokens"].describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))

    def show_common_n_grams(self, n=3, show_count=20):
        """
         Show the most common n - grams for each reviews type.

         @param n - number of words to use for ngrams
         @param show_count - number of words to show per type
        """
        # prints the most common ngrams for the given type
        for reviews_type, df_reviews in [("POSITIVE", self.df_positive_reviews), ("NEGATIVE", self.df_negative_reviews)]:
            text_ngrams = [ngram for review_text in df_reviews[self.reviews_col] for ngram in ngrams(review_text.split(), n)]
            most_common_ngrams = Counter(text_ngrams).most_common(show_count)
            print('-'*50)
            print(f"most common {n}-grams for {reviews_type}:")
            print('-'*50)
            print("n_grams -- occurence_counts")
            # Print the most common n grams and occurence counts.
            for n_grams, occurence_counts in most_common_ngrams:
                print(n_grams, "--", occurence_counts)

class EDA_medical_appointment:

    def __init__(self, df_medical_appointment_features, target_labels) -> None:
        """
         Initialize the class.
         
         @param df_medical_appointment_features - pandas DataFrame with the features
         @param target_labels - np.array of target labels 
        """
        self.df_medical_appointment = df_medical_appointment_features
        self.target_labels = target_labels
        self.numeric_feature_columns = ['Age']
        self.categorical_feature_columns = ['Gender', 'Neighbourhood', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received']

    def describe_general_stats(self):
        """
         Describe general stats and visualize them
        """
        print("-"*50)
        print("Checking feature datatypes and null values")
        print("-"*50, end='\n\n')
        display(self.df_medical_appointment.info())
        print("-"*50)
        print("Describe continuous numerical features")
        print("-"*50)
        display(self.df_medical_appointment[self.numeric_feature_columns].describe())
        self.df_medical_appointment[self.numeric_feature_columns].plot(kind='hist')
        plt.show()
        print("-"*50)
        print("categorical features Probabilities")
        print("-"*50)
        # Plot Independent Probability of Categorical Variable
        fig, axs = plt.subplots(figsize=(len(self.categorical_feature_columns)*2, 5), ncols=len(self.categorical_feature_columns), sharey=True)
        for cat_var, ax in zip(self.categorical_feature_columns, axs):
            # taking Most Frequent 10 Neighbourhoods to make the plot readable
            if cat_var=='Neighbourhood':
                cat_var_prob = self.df_medical_appointment[cat_var].value_counts().iloc[:10] / len(self.df_medical_appointment)
                cat_var = "Most Frequent 10 Neighbourhoods"
            else:
                cat_var_prob = self.df_medical_appointment[cat_var].value_counts() / len(self.df_medical_appointment)
            cat_var_prob.plot(kind="bar", ax=ax)
            ax.set_xlabel(cat_var)
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            ax.grid(axis='y')
        plt.suptitle(f"Independent Probability of Categorical Variable")
        plt.tight_layout()
        plt.show()

    def stats_per_unique_patient(self):
        """
         Plot stats of visits per unique patient for each patient.
        """
        df_medical_appointment_groupby_patient = self.df_medical_appointment.groupby('PatientID')
        df_medical_appointment_groupby_patient.count().sort_values(by=['AppointmentID'], ascending=False)['AppointmentID'].plot(kind='hist', density=True, bins=range(100))
        plt.xlabel('N Visits per unique patient')
        plt.title('Density Histogram of visits per unique patient')
        plt.show()
        n_patients_less_than_3_visits = len(df_medical_appointment_groupby_patient.count().loc[df_medical_appointment_groupby_patient.count()['AppointmentID']<3])
        df_patient_last_appointment = df_medical_appointment_groupby_patient.last()

        print("-"*50)
        print("-"*50)
        print(f"n_unique_patients = {len(df_patient_last_appointment)}")
        print(f"n_patients_less_than_3_visits = {n_patients_less_than_3_visits}")
        print(f"percent_patients_less_than_3_visits = {round(100*n_patients_less_than_3_visits/len(df_medical_appointment_groupby_patient.count()))}%")
        print("-"*50)
        print("-"*50)

        fig, axs = plt.subplots(figsize=(len(self.categorical_feature_columns)*2, 5), ncols=len(self.categorical_feature_columns), sharey=True)
        for cat_var, ax in zip(self.categorical_feature_columns, axs):
            # taking Most Frequent 10 Neighbourhoods to make the plot readable
            if cat_var=='Neighbourhood':
                cat_var_counts = df_patient_last_appointment[cat_var].value_counts().iloc[:10] / len(df_patient_last_appointment)
                cat_var = "Most Frequent 10 Neighbourhoods"
            else:
                cat_var_counts = df_patient_last_appointment[cat_var].value_counts() / len(df_patient_last_appointment)
            cat_var_counts.plot(kind="bar", ax=ax)
            ax.set_xlabel(cat_var)
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            ax.grid(axis='y')
        plt.suptitle(f"Independent Probability of Categorical Variable of Unique Patients")
        plt.tight_layout()
        plt.show()


    def visulaize_class_distribution(self):
        """
         Visualize the class distribution of the target data.
        """
        pd.Series(self.target_labels).value_counts().plot(kind="bar")
        plt.title("Class Distribution\n(0=Show, 1=No Show)")
        plt.ylabel("Count")
        plt.show()

    def visulaize_no_show_prob_per_variable(self):
        """
         Visualize the Probability of Feature given No-Show, i.e., P(Feature|no_show)
        """
        vars_categorical = self.categorical_feature_columns[:]
        vars_categorical.remove("Neighbourhood")
        fig, axs = plt.subplots(figsize=(len(vars_categorical)*2, 5), ncols=len(vars_categorical), sharey=True)
        df_no_show = self.df_medical_appointment.loc[self.target_labels==1]
        # Plots the conditional probability of each categorical variable.
        for cat_var, ax in zip(vars_categorical, axs):
            cat_var_prob_no_show = df_no_show[cat_var].value_counts() / len(df_no_show)
            cat_var_prob_no_show.plot(kind="bar", ax=ax)
            ax.set_xlabel(cat_var)
            ax.set_ylabel("probability")
            ax.set_ylim(0, 1)
            ax.grid(axis='y')
        plt.suptitle(f"Conditional Probability of Feature given No-Show\nP(Feature|no_show)")
        plt.tight_layout()
        plt.show()

        