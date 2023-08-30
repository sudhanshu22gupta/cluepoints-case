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
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from IPython.display import clear_output

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

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        # self.embeddings = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden_size * 1, output_size)  # *2 due to bidirectional
        
    def forward(self, x):
        # x = torch.dstack([x, x])
        # print(x.shape)
        # x = self.embeddings(x)
        print(x.shape)
        _, (h_n, _) = self.lstm(x)
        print(h_n.shape, h_n)
        # h_n_concat = torch.cat((h_n[-2, :], h_n[-1, :]), dim=-1)
        out = self.fc(h_n)  # Get the last time step's output
        return out

class BiLSTMClassifierWrapper(BinaryClassifier):
    def __init__(self, input_size, hidden_size, output_size, batch_size, num_epochs, lr_init):
        self.model = BiLSTMClassifier(input_size, hidden_size, output_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_init)
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def train_data_loader(self, X_train, y_train):
        X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        train_dataset = TensorDataset(X_train, y_train)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def fit_classifier(self, X_train, y_train):
        self.model.train()
        data_loader = self.train_data_loader(X_train, y_train)
        for epoch in range(self.num_epochs):
            for inputs, labels in data_loader:
                print(inputs.shape, labels.shape)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                print(outputs, labels)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
    
    def predict_classifier(self, X):
        self.model.eval()
        X = torch.tensor(X.toarray(), dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
        return predicted    

class DistilBERTClassifier(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.classifier = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=2,
            # seq_classif_dropout=0.3,
        )
        self.optimizer = Adam(self.parameters(), lr=3e-6)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_input(self, X_train, y_train, max_len=512):
        # create tokenized data
        self.preprocessor = preprocMovieReview(X_train.values)
        input_ids, attention_masks = self.preprocessor.distilbert_text_sanitization_pipeline(max_len=max_len)
        # convert the labels into tensors.
        labels = torch.tensor(y_train, dtype=torch.long)
        return TensorDataset(input_ids, attention_masks, labels)

    def fit_classifier(self, X_train, y_train, X_val, y_val, batch_size=8, n_epochs=2):

        self.dl_params = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 0
            }
        train_dataset = self.preprocess_input(X_train, y_train)
        val_dataset = self.preprocess_input(X_val, y_val)
        self.dataloader_train = DataLoader(train_dataset, **self.dl_params)
        self.dataloader_val = DataLoader(val_dataset, **self.dl_params)

        for epoch_num in range(n_epochs):
            self.train()
            train_loss = 0
            for step_num, train_data_batch in enumerate(self.dataloader_train):
                token_ids, masks, labels = tuple(t.to(self.device) for t in train_data_batch)
                model_output = self.classifier.forward(input_ids=token_ids, attention_mask=masks, labels=labels)
                
                batch_loss = model_output.loss
                train_loss += batch_loss.item()

                self.zero_grad()
                batch_loss.backward()

                nn.utils.clip_grad_norm_(parameters=self.classifier.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                clear_output(wait=True)
                print('Epoch: ', epoch_num + 1)
                print("\r" + "{0}/{1} loss: {2} ".format(step_num, len(self.dataloader_train), train_loss / (step_num + 1)))

    def predict_classifier(self, X_test, y_test):
        test_dataset = self.preprocess_input(X_test, y_test)
        self.dataloader_test = DataLoader(test_dataset, **self.dl_params)
        self.eval()
        pred_probs = []
        with torch.no_grad():
            for step_num, batch_data in enumerate(self.dataloader_test):
                token_ids, masks, labels = tuple(t.to(self.device) for t in batch_data)
                model_output = self.classifier.forward(input_ids=token_ids, attention_mask=masks, labels=labels)
                numpy_logits = model_output.logits.cpu().detach().numpy()
                pred_probs.extend(list(numpy_logits))
                clear_output(wait=True)
                print("\r" + "{0}/{1}".format(step_num, len(self.dataloader_test)))
        return (np.hstack(pred_probs) > 0.5).astype(int)

    def evaluate_classifier(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        # print(f"precision={precision}\nrecall={recall}\nf1={f1}\naccuracy={accuracy}")
        return precision, recall, f1, accuracy