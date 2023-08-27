import pandas as pd

def train_val_test_split(df, feature_cols, target_col, test_percent: float, val_percent: float=0.0):

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

    if val_percent:
        df_in_val = df_in
        X_val, y_val = df_in_val[feature_cols], df_in_val[target_col].values
    else:
        X_val, y_val = pd.Series(), pd.Series()
    
    return X_train, y_train, X_test, y_test, X_val, y_val
