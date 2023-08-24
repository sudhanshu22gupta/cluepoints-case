from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class BinaryClassifier:

    def __init__(self) -> None:
        pass

    def evaluate_classifier(self, y_true, y_pred):

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        print(f"{precision=}\n{recall=}\n{f1=}\n{accuracy=}")

class LogisticRegressionClassifier(BinaryClassifier):

    def __init__(self) -> None:
        self.classifier = LogisticRegression(C=1, penalty='l2', max_iter=1000, random_state=22)

    def fit_classifier(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)
    
    def predict_classifier(self, X):
        return self.classifier.predict(X)

# class BiLSTMClassifier(BinaryClassifier):

#     def __init__(self, **kwargs) -> None:
#         self.classifier = LSTM(**kwargs)

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 due to bidirectional
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Get the last time step's output
        return out

class BiLSTMClassifierWrapper:
    def __init__(self, input_size, hidden_size, output_size, batch_size, num_epochs, lr_init):
        self.model = BiLSTMClassifier(input_size, hidden_size, output_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_init)
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def train_data_loader(self, X_train, y_train):
        train_dataset = TensorDataset(X_train, y_train)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def fit_classifier(self, X_train, y_train):
        self.model.train()
        X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        data_loader = self.train_data_loader(X_train, y_train)
        for epoch in range(self.num_epochs):
            for inputs, labels in data_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
    
    def predict_classifier(self, X):
        self.model.eval()
        # X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
        return predicted    

class DistilBERTClassifier(BinaryClassifier):

    def __init__(self, **kwargs) -> None:
        self.classifier = None

    def fit_classifier(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)
    
    def predict_classifier(self, X):
        return self.classifier.predict(X)
