from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from xgboost import XGBClassifier
from transformers import TFBertForSequenceClassification
from simpletransformers.classification import ClassificationModel

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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

class MLPClf(BinaryClassifier):
    def __init__(self) -> None:
        self.classifier = MLPClassifier(warm_start=False, max_iter=1000, verbose=True)

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

class DistilBERTClassifier(BinaryClassifier):
    def __init__(self) -> None:
        # self.classifier = TFBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

        self.classifier = ClassificationModel(
            "distilbert", 
            "./data/distilbert-base-uncased",
            num_labels=2,
            reprocess_input_data=True,
            fp16=True,
            num_train_epochs=5,
            use_cuda=False,
        )

        print(self.classifier.summary())

    def fit_classifier(self, X_train, y_train):
        self.classifier.train_model(X_train, y_train)

    def fit_classifier(self, X_train, y_train):
        self.classifier.train_model(X_train, y_train)
