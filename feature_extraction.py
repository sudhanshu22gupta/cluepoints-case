import numpy as np
import pandas as pd
from itertools import islice

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn.utils import parallel_backend
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from scipy.cluster import hierarchy as hc
from collections import defaultdict
import matplotlib.pyplot as plt
from classification import RandomForestClassifier

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

class textVectorizer:

    def __init__(self, vectorizer_type, vocab_size=10_000) -> None:
        
        self.vocab_size = vocab_size
        if vectorizer_type == 'tfidf':
            self.initialize_tfidf_vectorizer()
        elif vectorizer_type == 'count':
            self.initialize_count_vectorizer()
        else:
            raise NotImplementedError
    
    def initialize_tfidf_vectorizer(self):
        self.vectorizer = TfidfVectorizer(
            min_df=0.0, 
            max_df=1.0, 
            use_idf=True, 
            ngram_range=(1,3), 
            max_features=self.vocab_size,
        )

    def initialize_count_vectorizer(self):
        self.vectorizer = CountVectorizer(
            min_df=0.0, 
            max_df=1.0, 
            binary=False, 
            ngram_range=(1,3), 
            max_features=self.vocab_size,
        )

    def apply_transform_train(self, reviews):
        return self.vectorizer.fit_transform(reviews)

    def apply_transform_test(self, reviews):
        return self.vectorizer.transform(reviews)

def truncate_or_pad_sequences(sequences, max_sequence_length, padding_value=0):
    """
    Truncate or pad sequences to a specified maximum sequence length.
    """
    truncated_padded_sequences = []
    for sequence in sequences:
        if len(sequence) > max_sequence_length:
            # Truncate if sequence is longer than max_sequence_length
            truncated_sequence = sequence[:max_sequence_length]
        else:
            # Pad if sequence is shorter than max_sequence_length
            padding_length = max_sequence_length - len(sequence)
            padded_sequence = np.pad(sequence, (0, padding_length), mode='constant', constant_values=padding_value)
            truncated_sequence = padded_sequence
        truncated_padded_sequences.append(truncated_sequence)
    return np.dstack(truncated_padded_sequences)

class GloveVectorizer:
    def __init__(self, embedding_dim, vocab_size, max_sequence_length, glove_emb_path):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.glove_emb_path = glove_emb_path
        self.word_to_index = {}
        self.embedding_matrix = None

        self.load_glove_embeddings()

    def load_glove_embeddings(self):
        # Initialize word_to_index and embedding_matrix
        self.word_to_index["<PAD>"] = 0  # Reserved for padding
        self.embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))

        # Load GloVe embeddings
        with open(self.glove_emb_path, 'r', encoding='utf-8') as glove_file:
            for index, line in enumerate(glove_file):
                if index >= self.vocab_size - 1:
                    break

                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')

                # Ensure the vector has the expected dimension
                if len(vector) == self.embedding_dim:
                    self.word_to_index[word] = index + 1  # Start index from 1, reserve 0 for padding
                    self.embedding_matrix[index + 1] = vector

    def text_to_sequence(self, text):
        words = text.split()
        sequence = [self.word_to_index.get(word, 0) for word in words][:self.max_sequence_length]
        while len(sequence) < self.max_sequence_length:
            sequence.append(0)  # Pad sequence if it's shorter than max_sequence_length
        return sequence

    def vectorize_text(self, reviews):
        review_vectors = []
        for review_text in reviews:
            sequence = self.text_to_sequence(review_text)
            review_vectors.append(self.embedding_matrix[sequence])
        return np.transpose(np.dstack(review_vectors), (2, 0, 1))

class featuresMedicalAppointment:

    def __init__(self) -> None:
        self.ohes = {}
        self.minmax_scalers = {}
        self.infrequent_values = {}

    def feat_minmax_norm_train(self, df_medical_appointment, columns_to_scale):
        for column in columns_to_scale:
            try:
                assert column in df_medical_appointment.columns
            except AssertionError:
                raise AssertionError(f"Specified column: {column} not in dataframe")
            minmax_scaler = MinMaxScaler()
            df_medical_appointment[column] = minmax_scaler.fit_transform(df_medical_appointment[column].values.reshape(-1, 1))
            self.minmax_scalers[column] = minmax_scaler
        return df_medical_appointment

    def feat_minmax_norm_test(self, df_medical_appointment, columns_to_scale):
        for column in columns_to_scale:
            try:
                assert column in df_medical_appointment.columns
            except AssertionError:
                raise AssertionError(f"Specified column: {column} not in dataframe")
            try:
                minmax_scaler = self.minmax_scalers[column]
            except KeyError:
                raise Exception(f"Min-Max Scaler not trained for {column}")
            df_medical_appointment[column] = minmax_scaler.fit_transform(df_medical_appointment[column].values.reshape(-1, 1))
            self.minmax_scalers[column] = minmax_scaler
        return df_medical_appointment

    def feat_n_hours_scheduled_before(self, df_medical_appointment, scheduled_after_appointment_strategy):
        df_medical_appointment['n_hours_scheduled_before'] = (pd.to_datetime(df_medical_appointment['AppointmentDay']) - pd.to_datetime(df_medical_appointment['ScheduledDay'])).apply(lambda x: round(x.total_seconds()/3600))
        # Schedule Datetime > Appointment Datetime ==0> probably entry mistakes ==> replace with nan or 0
        if isinstance(scheduled_after_appointment_strategy, (int, float)):
            df_medical_appointment.loc[df_medical_appointment['n_hours_scheduled_before'] < 0, 'n_hours_scheduled_before'] = scheduled_after_appointment_strategy
        elif scheduled_after_appointment_strategy.lower()=='drop':
            df_medical_appointment = df_medical_appointment.loc[~(df_medical_appointment['n_hours_scheduled_before'] < 0)]
            df_medical_appointment.reset_index(drop=True, inplace=True)
        else:
            raise Exception("invalid value for scheduled_after_appointment_strategy")
        return df_medical_appointment
    
    def feat_appointment_date(self, df_medical_appointment):
        df_medical_appointment['AppointmentDay'] = pd.to_datetime(df_medical_appointment['AppointmentDay'])
        df_medical_appointment['day_of_month'] = df_medical_appointment['AppointmentDay'].apply(lambda x: x.day)
        df_medical_appointment['day_of_week'] = df_medical_appointment['AppointmentDay'].apply(lambda x: x.day_name())
        # all hour of day values are 0
        # df_medical_appointment['hour_of_day'] = df_medical_appointment['AppointmentDay'].apply(lambda x: x.hour)
        return df_medical_appointment

    def feat_categorical_to_one_hot_encoding_train(self, df_medical_appointment, infrequent_threshold=20):
        self.categorical_feature_columns = ['Gender', 'Neighbourhood', 'day_of_week']
        self.infrequent_threshold = infrequent_threshold
        for column in self.categorical_feature_columns:
            column_value_counts = df_medical_appointment[column].value_counts()
            infrequent_values = column_value_counts.loc[column_value_counts<infrequent_threshold].index.values
            df_medical_appointment[column].replace(infrequent_values, 'INFREQUENT', inplace=True)
            self.infrequent_values[column] = infrequent_values
            print(column, "-> n_unique:", len(df_medical_appointment[column].unique()))
            ohe = OneHotEncoder(handle_unknown='ignore')
            encoded_values = ohe.fit_transform(df_medical_appointment[column].values.reshape(-1,1)).toarray()
            encoded_labels = np.hstack(ohe.categories_)
            df_ohe = pd.DataFrame(encoded_values, columns=encoded_labels, dtype='int')
            df_ohe.rename(columns={ohe_col: f"{column}_{ohe_col}" for ohe_col in df_ohe.columns}, inplace=True)
            df_medical_appointment.drop(columns=column, inplace=True)
            df_medical_appointment.reset_index(drop=True, inplace=True)
            df_medical_appointment = pd.concat([df_medical_appointment, df_ohe], axis=1)
            self.ohes[column] = ohe
        return df_medical_appointment
    
    def feat_categorical_to_one_hot_encoding_test(self, df_medical_appointment):
        for column in self.categorical_feature_columns:
            infrequent_values = self.infrequent_values[column]
            df_medical_appointment[column].replace(infrequent_values, 'INFREQUENT', inplace=True)
            print(column, "-> n_unique:", len(df_medical_appointment[column].unique()))
            try:
                ohe = self.ohes[column]
            except KeyError:
                raise Exception(f"One Hot Encoder not trained for {column}")
            encoded_values = ohe.transform(df_medical_appointment[column].values.reshape(-1,1)).toarray()
            encoded_labels = np.hstack(ohe.categories_)
            df_ohe = pd.DataFrame(encoded_values, columns=encoded_labels, dtype='int')
            df_ohe.rename(columns={ohe_col: f"{column}_{ohe_col}" for ohe_col in df_ohe.columns}, inplace=True)
            df_medical_appointment.drop(columns=column, inplace=True)
            df_medical_appointment.reset_index(drop=True, inplace=True)
            df_medical_appointment = pd.concat([df_medical_appointment, df_ohe], axis=1)
        return df_medical_appointment

    def plot_optimal_PCA_components(self, df_medical_appointment):
        
        max_components = len(df_medical_appointment.columns)
        # Fit PCA with different numbers of components
        explained_variance = []
        for n in range(1, max_components + 1):
            pca = PCA(n_components=n)
            pca.fit(df_medical_appointment)
            explained_variance.append(np.sum(pca.explained_variance_ratio_))

        # Plot explained variance ratio
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_components + 1), explained_variance, marker='x')
        plt.title('Explained Variance Ratio vs. Number of Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance Ratio')
        plt.grid(True)
        plt.show()

    def feat_PCA_train(self, df_medical_appointment, n_components):
        self.pca = PCA(n_components=n_components)
        pca_transfomed = self.pca.fit_transform(df_medical_appointment)
        return pd.DataFrame(pca_transfomed, columns=[f"feature_{i}" for i in range(n_components)])

    def feat_PCA_test(self, df_medical_appointment):
        pca_transfomed = self.pca.transform(df_medical_appointment)
        return pd.DataFrame(pca_transfomed, columns=[f"feature_{i}" for i in range(self.pca.n_components)])

def feature_selection_permutation_importance(df_features, target, n_repeats=8, n_jobs=8, plot=True):

    clf = RandomForestClassifier()
    clf.fit(df_features, target)
    with parallel_backend('multiprocessing', n_jobs=n_jobs):
        perm_imp = permutation_importance(
            estimator = clf,
            X = df_features, 
            y = target, 
            n_repeats = n_repeats,
            n_jobs = n_jobs,
        )
    
    df_feature_importances = pd.DataFrame(list(zip(perm_imp.importances_mean, df_features.columns)), columns=['coef','feature'])
    df_feature_importances = df_feature_importances.sort_values(by='coef', ascending=False)
    df_feature_importances.reset_index(drop=True, inplace=True)

    if plot:
        N_FEATS_PLOT = len(df_feature_importances)
        plt.figure(figsize=(30, 40))
        plt.barh(list(range(len(df_feature_importances)))[:N_FEATS_PLOT][::-1], df_feature_importances['coef'].values[:N_FEATS_PLOT],)
        plt.yticks(list(range(len(df_feature_importances)))[:N_FEATS_PLOT][::-1], df_feature_importances['feature'].values[:N_FEATS_PLOT])
        plt.xticks(np.arange(0, 0.15, 0.005))
        plt.grid(axis='x')
        plt.tight_layout()
        plt.show()

    return df_feature_importances

def get_high_perm_imp_feat(df_feature_importances, features_list, threshold_importance=None):
    df_ = df_feature_importances.copy()
    df_ = df_.loc[df_.index.isin(features_list)]
    df_ = df_.sort_values(by='coef', ascending=False)
    if threshold_importance:
        if df_.iloc[0]['coef'] < threshold_importance:
            return None
    return df_.index.values[0]

def feature_selection_hierarchical_clustering(df_features, threshold_clustering, df_feature_importances, threshold_importance=None, plot=True):

    corr = np.round(spearmanr(df_features).correlation, 4)
    corr_condensed = hc.distance.squareform(1 - corr)
    z = hc.linkage(corr_condensed, method='average')
    features_list = df_features.columns
    if plot:
        plt.figure(figsize=(50,40))
        hc.dendrogram(z, labels=features_list, orientation='top', leaf_font_size=16)
        plt.yticks(np.arange(0, 1.1, 0.05))
        plt.grid(axis='y')
        plt.show()

    cluster_ids = hc.fcluster(z, t=threshold_clustering, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)

    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    
    assert isinstance(df_feature_importances, pd.DataFrame)
    selected_features = [get_high_perm_imp_feat(df_feature_importances, feature_id, threshold_importance=threshold_importance) for feature_id in cluster_id_to_feature_ids.values()]

    selected_features = [features_list[x] for x in selected_features if x is not None]
    print(f'{len(selected_features)} Selected features:\n', selected_features)
    df_selected_features = df_feature_importances.loc[df_feature_importances['feature'].isin(selected_features)]

    return df_selected_features
