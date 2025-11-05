"""
Example usage of the stateless Materials Revenue Prediction API

Run the API: python api.py
Run this example: python api_example.py
"""

import requests
import json
import base64

API_BASE_URL = "http://localhost:8000"

def example_train():
    """Train a model by uploading CSV data"""
    print("=" * 60)
    print("EXAMPLE 1: Training a model")
    print("=" * 60)
    
    with open('./data/train.txt', 'rb') as f:
        files = {'file': ('train.txt', f, 'text/plain')}
        response = requests.post(f"{API_BASE_URL}/train", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print("\n✓ Model trained successfully!")
        print(f"\nMetrics:")
        for metric, value in result['metrics'].items():
            if metric == 'r2':
                print(f"  {metric.upper()}: {value:.4f}")
            else:
                print(f"  {metric.upper()}: ${value:,.2f}")
        
        model_data = base64.b64decode(result['model_data'])
        encoder_data = base64.b64decode(result['encoder_data'])
        
        with open('temp_model.txt', 'wb') as f:
            f.write(model_data)
        with open('temp_encoder.pkl', 'wb') as f:
            f.write(encoder_data)
        
        print(f"\n✓ Saved model to temp_model.txt ({len(model_data)} bytes)")
        print(f"✓ Saved encoder to temp_encoder.pkl ({len(encoder_data)} bytes)")
        
        return True
    else:
        print(f"\n✗ Training failed: {response.text}")
        return False

def example_predict():
    """Predict revenues by uploading model, encoder, and materials"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Predicting revenues")
    print("=" * 60)
    
    materials = [
        {"date": "2024-06-01", "price": 1800.00, "type": "window", "distance": 60.0,
         "size_length": 25.0, "size_width": 35.0, "size_height": 6.0},
        {"date": "2024-06-15", "price": 2500.00, "type": "flooring", "distance": 45.0,
         "size_length": 70.0, "size_width": 50.0, "size_height": 2.0},
        {"date": "2024-07-01", "price": 1000.00, "type": "drywall", "distance": 100.0,
         "size_length": 40.0, "size_width": 30.0, "size_height": 1.5},
    ]
    
    files = {
        'model_file': ('model.txt', open('temp_model.txt', 'rb'), 'application/octet-stream'),
        'encoder_file': ('encoder.pkl', open('temp_encoder.pkl', 'rb'), 'application/octet-stream'),
    }
    
    data = {'materials': json.dumps(materials)}
    
    response = requests.post(f"{API_BASE_URL}/predict", files=files, data=data)
    
    files['model_file'][1].close()
    files['encoder_file'][1].close()
    
    if response.status_code == 200:
        predictions = response.json()['predictions']
        print("\n✓ Predictions successful!")
        print(f"\n{'Type':<12} {'Price':<10} {'Distance':<10} {'Size (LxWxH)':<20} {'Predicted Revenue':<20}")
        print("-" * 80)
        for material, prediction in zip(materials, predictions):
            size_str = f"{material['size_length']:.1f}x{material['size_width']:.1f}x{material['size_height']:.1f}"
            print(f"{material['type']:<12} ${material['price']:<9.2f} {material['distance']:<10.1f} {size_str:<20} ${prediction:>18,.2f}")
    else:
        print(f"\n✗ Prediction failed: {response.text}")

if __name__ == "__main__":
    print("\nStateless Materials Revenue Prediction API - Example Usage\n")
    
    try:
        if example_train():
            example_predict()
        
        print("\n" + "=" * 60)
        print("Examples completed!")
        print("=" * 60)
        print("\nNote: This API is completely stateless!")
        print("The model and encoder are passed with every prediction request.")
        print("\nAPI Documentation: http://localhost:8000/docs")
        
        import os
        os.remove('temp_model.txt')
        os.remove('temp_encoder.pkl')
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API server.")
        print("  Please make sure the API is running: python api.py")
    except FileNotFoundError:
        print("\n✗ Error: Training data file not found.")
        print("  Make sure ./data/train.txt exists")
