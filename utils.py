import pandas as pd
import matplotlib.pyplot as plt

def train_val_test_split(df, feature_cols, target_col, test_percent: float, val_percent: float=0.0):
    """
     Split the dataset into training validation and test sets.
     
     @param df - Dataframe containing the features to train
     @param feature_cols - Column containing the features to train
     @param target_col - Column containing the target labels to test
     @param test_percent - Percent of the dataset to split for training
     @param val_percent - Percent of the dataset to split for validation
     
     @return A tuple containing the training set, test set, validation set
    """
    
    assert test_percent+val_percent < 100
    train_percent = 100 - (test_percent+val_percent)
    total_size = len(df)

    # shuffle the dataset
    df_in = df.copy().sample(total_size)
    
    df_in_train = df_in.iloc[:int((train_percent*total_size)/100)]
    X_train, y_train = df_in_train[feature_cols], df_in_train[target_col].values
    df_in = df_in.iloc[int((train_percent*total_size)/100):]

    df_in_test = df_in.iloc[:int((test_percent*total_size)/100)]
    X_test, y_test = df_in_test[feature_cols], df_in_test[target_col].values
    df_in = df_in.iloc[int((test_percent*total_size)/100):]

    # if val percent not defined, empty series returned for validation sets
    if val_percent:
        df_in_val = df_in
        X_val, y_val = df_in_val[feature_cols], df_in_val[target_col].values
    else:
        X_val, y_val = pd.Series(dtype='float16'), pd.Series(dtype='float16').values
    
    return X_train, y_train, X_test, y_test, X_val, y_val


def compare_performances_across_classifiers(df_metrics):
    """
    Plot the comparision of performance across classifer for a task

     @param df_metrics: Dataframe with performance metircs for each classifer
    """
    metrics_to_compare = ['precision', 'recall', 'f1', 'accuracy']
    df_metrics.reset_index(drop=True, inplace=True)
    plt.figure(figsize=(12,8))
    df_metrics[metrics_to_compare].plot(ax=plt.gca())
    plt.xticks(df_metrics.index, df_metrics['classifier'].values, rotation=90)
    plt.legend()
    plt.title("Compraring metrics across classifiers")
    plt.xlabel("Classifier")
    plt.ylabel("Metric Score")
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.show()