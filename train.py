from __future__ import division
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import io

def train_model_from_data(training_data_df):
    """
    Train a model from a DataFrame containing materials and revenues.
    
    Args:
        training_data_df: DataFrame with columns: date, revenue, price, type, distance, size_length, size_width, size_height
        
    Returns:
        tuple: (model_bytes, encoder_bytes, metrics_dict)
    """
    df = training_data_df.copy()
    
    df['date_numeric'] = pd.to_datetime(df['date']).astype(int) / 10**9 / 86400
    
    label_encoder = LabelEncoder()
    df['type_encoded'] = label_encoder.fit_transform(df['type'])
    
    y = df['revenue']
    X = df.drop(['revenue', 'date', 'type'], axis=1)
    
    df_test = df.sample(frac=0.2, random_state=42)
    df_train = df.drop(df_test.index)
    
    y_train = df_train['revenue']
    X_train = df_train.drop(['revenue', 'date', 'type'], axis=1)
    y_test = df_test['revenue']
    X_test = df_test.drop(['revenue', 'date', 'type'], axis=1)
    
    lgb_train = lgb.Dataset(X_train, y_train)
    
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'num_leaves': 63,
        'num_trees': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=lgb_train)
    
    y_pred_test = gbm.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    model_str = gbm.model_to_string()
    model_bytes = model_str.encode('utf-8')
    
    encoder_bytes = pickle.dumps(label_encoder)
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }
    
    return model_bytes, encoder_bytes, metrics