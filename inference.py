from __future__ import division
import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle

def predict_revenue(properties):
    """
    Predict revenue for given property attributes.
    
    Args:
        properties: List of dictionaries containing property attributes.
                   Each item should have: date, price, type, distance, size_length, size_width, size_height
                   
    Returns:
        numpy array of predicted revenues
    """
    gbm = lgb.Booster(model_file='model.txt')
    
    with open('type_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    df = pd.DataFrame(properties)
    df['date_numeric'] = pd.to_datetime(df['date']).astype(int) / 10**9 / 86400
    df['type_encoded'] = label_encoder.transform(df['type'])
    X = df.drop(['date', 'type'], axis=1)
    
    predictions = gbm.predict(X)
    return predictions  

if __name__ == '__main__':
    print('Load saved model...')
    
    example_properties = [
        {'date': '2024-01-15', 'price': 1500.00, 'type': 'window', 'distance': 50.0, 'size_length': 20.0, 'size_width': 30.0, 'size_height': 5.0},
        {'date': '2024-02-20', 'price': 2000.00, 'type': 'roofing', 'distance': 75.0, 'size_length': 50.0, 'size_width': 40.0, 'size_height': 10.0},
        {'date': '2024-03-10', 'price': 800.00, 'type': 'door', 'distance': 120.0, 'size_length': 10.0, 'size_width': 15.0, 'size_height': 3.0},
        {'date': '2024-04-05', 'price': 3500.00, 'type': 'lumber', 'distance': 30.0, 'size_length': 80.0, 'size_width': 60.0, 'size_height': 20.0},
        {'date': '2024-05-12', 'price': 1200.00, 'type': 'insulation', 'distance': 90.0, 'size_length': 30.0, 'size_width': 25.0, 'size_height': 8.0},
    ]
    
    print(f'\nPredicting revenue for {len(example_properties)} materials...')
    predictions = predict_revenue(example_properties)
    
    print('\nPredictions:')
    print(f'{"Date":<15} {"Price":<10} {"Type":<12} {"Distance":<10} {"L x W x H":<20} {"Predicted Revenue":<20}')
    print('-' * 95)
    for prop, pred in zip(example_properties, predictions):
        size_str = f"{prop['size_length']:.1f} x {prop['size_width']:.1f} x {prop['size_height']:.1f}"
        print(f"{prop['date']:<15} ${prop['price']:<9.2f} {prop['type']:<12} {prop['distance']:<10.2f} {size_str:<20} ${pred:<19.2f}")
    
    print('\nMaterial types: window, door, roofing, siding, insulation, flooring, drywall, lumber')
