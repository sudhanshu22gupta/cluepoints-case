from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from xgboost import XGBClassifier
from transformers import DistilBertForSequenceClassification
from skopt import BayesSearchCV

from preprocessing import preprocMovieReview
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Bidirectional, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryCrossentropy, Precision, Recall
from scipy.sparse import csr_matrix

import os
import traceback
from tqdm import tqdm
from datetime import datetime
from IPython.display import clear_output
import matplotlib.pyplot as plt

class BinaryClassifier:

    def __init__(self) -> None:
        pass

    def fit_classifier(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)
    
    def predict_classifier(self, X):
        return self.classifier.predict(X)

    def predict_proba_classifier(self, X):
        return self.classifier.predict_proba(X)

    def predict_at_threshold(self, X, clf_threshold):
        return (self.predict_proba_classifier(X)[:,1] > clf_threshold).astype(int)

    def evaluate_classifier(self, y_true, y_pred):

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        # print(f"precision={precision}\nrecall={recall}\nf1={f1}\naccuracy={accuracy}")

        return precision, recall, f1, accuracy

    def predict_per_threshold(self, X, y_true, threshold_values, plot=True):
        
        df_results = pd.DataFrame()
        # predict per threshold
        for clf_threshold in threshold_values:
            # classifer only
            y_pred = self.predict_at_threshold(X, clf_threshold)
            precision, recall, f1, accuracy = self.evaluate_classifier(y_true, y_pred)

            # record metrics
            df_results = df_results.append([{
                "threshold": clf_threshold, 
                "precision": precision, 
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
            }])
        df_results.reset_index(drop=True, inplace=True)
        
        if plot:
            df_results.plot(y=["precision", "recall", "f1", "accuracy"], x="threshold")

        return df_results

class LogisticRegressionClf(BinaryClassifier):
    def __init__(self) -> None:
        self.classifier = LogisticRegression(C=1, penalty='l2', max_iter=1000, random_state=22)

class RandomForestClf(BinaryClassifier):
    def __init__(self) -> None:
        self.classifier = RandomForestClassifier()

    def hyperparmater_tuning(self, X_train, y_train, param_space, n_iter=20, cv=5, n_jobs=8):
        self.bayes_search = BayesSearchCV(
            self.classifier,
            param_space,
            n_iter=n_iter,
            cv=cv,
            n_jobs=n_jobs,
        )
        self.bayes_search.fit(X_train, y_train)
        self.classifier = RandomForestClassifier(**self.bayes_search.best_params_)
        self.fit_classifier(X_train, y_train)

class MLPClf(BinaryClassifier):
    def __init__(self) -> None:
        self.classifier = MLPClassifier(
            hidden_layer_sizes = [512, 256, 64],
            warm_start=False, 
            max_iter=1000, 
            verbose=True,
        )

class XGBClf(BinaryClassifier):
    def __init__(self) -> None:
        self.classifier = XGBClassifier()

class BiLSTMClf:
    
    def __init__(
        self, 
        embedding_dim,
        vocab_size,
        max_sequence_length,
        HIDDEN_ACTIVATION,
        MAX_EPOCHS,
        LR_INIT,
        BATCH_SIZE,
        L2_REG_PENALTY,
        CALLBACKS,
        VERBOSITY_LEVEL,
        SAVE_DIR,
    ):
        self.HIDDEN_ACTIVATION = HIDDEN_ACTIVATION
        self.MAX_EPOCHS = MAX_EPOCHS
        self.LR_INIT = LR_INIT
        self.BATCH_SIZE = BATCH_SIZE
        self.L2_REG_PENALTY = L2_REG_PENALTY
        self.VERBOSITY_LEVEL = VERBOSITY_LEVEL
        self.SAVE_DIR = SAVE_DIR
        self.CALLBACKS = CALLBACKS
        self.EMBEDDING_DIM = embedding_dim
        self.VOCAB_SIZE = vocab_size
        self.MAX_SEQUENCE_LENGTH = max_sequence_length

        self.create_model_BiLSTM()
        self.compile_model()
        self.create_callbacks()

    def create_model_BiLSTM(self, ):
        
        # Initlaize Sequential Model
        self.model = tf.keras.Sequential()
        # Embedding layer
        self.model.add(Input(shape=(self.MAX_SEQUENCE_LENGTH, self.EMBEDDING_DIM)))
        # Bidirectional LSTM layer with dropout
        self.model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
        # Bidirectional LSTM layer with dropout
        self.model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
        # Global Max Pooling layer
        self.model.add(tf.keras.layers.GlobalMaxPooling1D())
        # Fully connected layer with dropout
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.2))  # Adjust dropout rate
        # Output layer for binary classification
        self.model.add(Dense(1, activation='sigmoid'))
        print(self.model.summary())

    def create_callbacks(
        self, 
        min_delta=1e-4,
        patience_es=20,
        patience_rlrop=10,
        factor_rlopl=0.1,
        tensorboard_histogram_freq=25,
    ):
        
        self.callbacks = []

        if "es" in self.CALLBACKS:
            es = EarlyStopping(
                monitor='val_loss', 
                min_delta=min_delta, 
                patience=patience_es, 
                verbose=1,
                restore_best_weights=True,
            )
            self.callbacks.append(es)

        if "rlrop" in self.CALLBACKS: 
            rlrop = ReduceLROnPlateau(
                monitor='val_loss', 
                factor=factor_rlopl, 
                patience=patience_rlrop, 
                verbose=1,
                min_delta=min_delta,
            )
            self.callbacks.append(rlrop)

        if "tensorboard" in self.CALLBACKS:
            tensorboard_logdir = os.path.join(self.SAVE_DIR)
            if not os.path.exists(tensorboard_logdir):
                os.makedirs(tensorboard_logdir)
            tensorboard_callback = TensorBoard(
                log_dir= tensorboard_logdir,
                histogram_freq=tensorboard_histogram_freq,
                )
            self.callbacks.append(tensorboard_callback)

    def compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=self.LR_INIT,),
            loss='binary_crossentropy',
            metrics=[
                BinaryCrossentropy(name='binary_crossentropy'),
                Precision(),
                Recall(),
            ]
        )

    def fit_classifier(self, X_train, y_train, X_val, y_val):
        try:
            self.history = self.model.fit(
                x=X_train,
                y=y_train,
                batch_size=self.BATCH_SIZE,
                epochs=self.MAX_EPOCHS,
                validation_data=(X_val, y_val),
                callbacks=self.callbacks,
                verbose=self.VERBOSITY_LEVEL,
            )
            # Save the entire model as a SavedModel.
            self.model.save(os.path.join(self.SAVE_DIR, f'{datetime.now().strftime("%Y%m%d%H%M%S")}_BiLSTM_classifier'))
            self.plot_history()
        except Exception:
            print("Error in training")
            traceback.print_exc()

    def plot_history(self):
        """
        Plot Metrics: Loss, Precision and Recall across epochs for both train and validation sets.
        """
        plot_metrics = ['loss', 'precision', 'recall']
        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(18, 6))
        for i, metric in enumerate(plot_metrics):
            ax = axs[i]

            metric_key = metric
            for key in self.history.history.keys():
                if metric in key:
                    metric_key = key
                    break
            ax.plot(self.history.history[f'{metric_key}'], label='train')
            ax.plot(self.history.history[f'val_{metric_key}'], label='val')
            ax.set_title(f'{metric}')
            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric)
            ax.set_ylim(-0.1, 1.1)
            ax.legend()
        
        plt.savefig(os.path.join(self.SAVE_DIR, f'{datetime.now().strftime("%Y%m%d%H%M%S")}_BiLSTM_classifier_Train_Val_Performance.png'))
        plt.show()

    def evaluate_classifier(self, y_true, y_pred):

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        return precision, recall, f1, accuracy

    def predict_classifier(self, X):
        return np.hstack(self.model.predict(X) > 0.5).astype(int)

class DistilBERTClassifier(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=2,
        )
        self.classifier.to(self.device)
        self.optimizer = AdamW(self.parameters(), lr=1e-3)

    def preprocess_input(self, X_train, y_train, max_len=512):
        # create tokenized data
        self.preprocessor = preprocMovieReview(X_train.values)
        input_ids, attention_masks = self.preprocessor.distilbert_text_sanitization_pipeline(max_len=max_len)
        # convert the labels into tensors.
        labels = torch.tensor(y_train, dtype=torch.long)
        return TensorDataset(input_ids, attention_masks, labels)

    def _train_step(self, train_data_batch):
        token_ids, masks, labels = tuple(t.to(self.device) for t in train_data_batch)
        model_output_train = self.classifier.forward(input_ids=token_ids, attention_mask=masks, labels=labels)
        batch_loss_train = model_output_train.loss
        self.train_loss += batch_loss_train.item()
        self.zero_grad()
        batch_loss_train.backward()
        nn.utils.clip_grad_norm_(parameters=self.classifier.parameters(), max_norm=1.0)
        self.optimizer.step()

    def _val_step(self, val_data_batch):
        token_ids, masks, labels = tuple(t.to(self.device) for t in val_data_batch)
        model_output_val = self.classifier.forward(input_ids=token_ids, attention_mask=masks, labels=labels)
        self.val_loss += model_output_val.loss.item()
        val_logits = model_output_val.logits.cpu().detach().numpy()
        preds = np.append(preds, np.argmax(val_logits, axis=1).flatten())


    def fit_classifier(self, X_train, y_train, X_val, y_val, batch_size=16, n_epochs=2):

        self.dl_params = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 0
            }
        train_dataset = self.preprocess_input(X_train, y_train)
        val_dataset = self.preprocess_input(X_val, y_val)
        self.dataloader_train = DataLoader(train_dataset, **self.dl_params)
        self.dataloader_val = DataLoader(val_dataset, **self.dl_params)
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
        }
        for epoch_num in range(n_epochs):

            self.train()
            self.train_loss = 0
            progress_bar = tqdm(total=len(self.dataloader_train), desc='Training Progress', position=0)
            for step_num, train_data_batch in enumerate(self.dataloader_train):
                # Training Step
                self._train_step(train_data_batch)
                # Update the progress bar
                progress_bar.set_postfix(epoch=epoch_num+1, train_loss=self.train_loss/(step_num+1))
                progress_bar.update()
            # Close the progress bar at end of epoch
            progress_bar.close()

            self.eval()
            self.val_loss = 0
            progress_bar = tqdm(total=len(self.dataloader_val), desc='Validation Progress', position=0)
            for step_num, val_data_batch in enumerate(self.dataloader_val):
                # Training Step
                self._val_step(val_data_batch)
                # Update the progress bar
                progress_bar.set_postfix(epoch=epoch_num+1, val_loss=self.val_loss/(step_num+1))
                progress_bar.update()
            # Close the progress bar at end of epoch
            progress_bar.close()

            # Record metrics
            self.history["epoch"].append(epoch_num+1)
            self.history["train_loss"].append(self.train_loss/(step_num+1))
            self.history["val_loss"].append(self.val_loss/(step_num+1))
        
        self.plot_history()
        torch.cuda.empty_cache()

    def plot_history(self):
        """
        Plot Metrics: Loss across epochs for both train and validation sets.
        """
        plt.figure(figsize=(6, 6))
        plt.plot(self.history['train_loss'], label='train')
        plt.plot(self.history['val_loss'], label='val')
        plt.title('Loss across Epochs')
        plt.xlabel('Epochs')
        plt.ylabel("Loss")
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.show()

    def predict_classifier(self, X_test, y_test):
        test_dataset = self.preprocess_input(X_test, y_test)
        self.dataloader_test = DataLoader(test_dataset, **self.dl_params)
        self.eval()
        preds = np.array([])
        with torch.no_grad():
            for step_num, batch_data in enumerate(self.dataloader_test):
                token_ids, masks, labels = tuple(t.to(self.device) for t in batch_data)
                model_output = self.classifier.forward(input_ids=token_ids, attention_mask=masks, labels=labels)
                numpy_logits = model_output.logits.cpu().detach().numpy()
                preds = np.append(preds, np.argmax(numpy_logits, axis=1).flatten())
        return preds

    def evaluate_classifier(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        # print(f"precision={precision}\nrecall={recall}\nf1={f1}\naccuracy={accuracy}")
        return precision, recall, f1, accuracy