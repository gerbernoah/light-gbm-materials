from __future__ import division
import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle

def predict_revenue_from_model(materials, model_bytes, encoder_bytes):
    """
    Predict revenue for given materials using provided model and encoder.
    Stateless function - does not read from disk.
    
    Args:
        materials: List of dictionaries containing property attributes.
                  Each item should have: date, price, type, distance, size_length, size_width, size_height
        model_bytes: Bytes of the trained LightGBM model
        encoder_bytes: Bytes of the pickled label encoder
        
    Returns:
        numpy array of predicted revenues
    """
    model_str = model_bytes.decode('utf-8')
    gbm = lgb.Booster(model_str=model_str)
    
    label_encoder = pickle.loads(encoder_bytes)
    
    df = pd.DataFrame(materials)
    df['date_numeric'] = pd.to_datetime(df['date']).astype(int) / 10**9 / 86400
    df['type_encoded'] = label_encoder.transform(df['type'])
    X = df.drop(['date', 'type'], axis=1)
    
    predictions = gbm.predict(X)