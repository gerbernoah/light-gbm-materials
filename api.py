from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import io

app = FastAPI(title="Materials Revenue Prediction API - Stateless")

class Material(BaseModel):
    date: str
    price: float
    type: str
    distance: float
    size_length: float
    size_width: float
    size_height: float

class PredictRequest(BaseModel):
    materials: List[Material]

class PredictResponse(BaseModel):
    predictions: List[float]

class TrainResponse(BaseModel):
    model_data: str
    encoder_data: str
    metrics: dict

@app.post("/train", response_model=TrainResponse)
async def train(file: UploadFile = File(...)):
    """
    Train a model from uploaded CSV data.
    Stateless endpoint - returns model and encoder as downloadable files.
    
    Upload a CSV file with columns: date, revenue, price, type, distance, size_length, size_width, size_height
    
    Returns:
        JSON with model_data (base64), encoder_data (base64), and metrics
    """
    try:
        from train import train_model_from_data
        import base64
        
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        required_cols = ['date', 'revenue', 'price', 'type', 'distance', 'size_length', 'size_width', 'size_height']
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(
                status_code=400, 
                detail=f"CSV must contain columns: {', '.join(required_cols)}"
            )
        
        model_bytes, encoder_bytes, metrics = train_model_from_data(df)
        
        return TrainResponse(
            model_data=base64.b64encode(model_bytes).decode('utf-8'),
            encoder_data=base64.b64encode(encoder_bytes).decode('utf-8'),
            metrics=metrics
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict", response_model=PredictResponse)
async def predict(
    materials: List[Material],
    model_file: UploadFile = File(...),
    encoder_file: UploadFile = File(...)
):
    """
    Predict revenues for materials using uploaded model and encoder.
    Stateless endpoint - requires model and encoder files with each request.
    
    Args:
        materials: JSON array of material objects
        model_file: Trained model file (from /train endpoint)
        encoder_file: Label encoder file (from /train endpoint)
        
    Returns:
        Array of predicted revenues
    """
    try:
        from inference import predict_revenue_from_model
        
        model_bytes = await model_file.read()
        encoder_bytes = await encoder_file.read()
        
        materials_list = [m.model_dump() for m in materials]
        predictions = predict_revenue_from_model(materials_list, model_bytes, encoder_bytes)
        
        return PredictResponse(predictions=predictions.tolist())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Materials Revenue Prediction API - Stateless",
        "endpoints": {
            "/train": "POST - Upload training CSV, returns model and encoder (multipart/form-data)",
            "/predict": "POST - Upload model, encoder, and materials JSON, returns predictions (multipart/form-data)",
            "/docs": "GET - Interactive API documentation"
        },
        "note": "This API is completely stateless - no data is stored on the server"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
