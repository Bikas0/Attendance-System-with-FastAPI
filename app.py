import os
import cv2
import pickle
import uvicorn
import uuid
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from utils import feature_extractor, extract_faces_mtcnn, faces_mtcnn, process_image

app = FastAPI()
router = APIRouter(prefix="/face_verify")

PKL_DIR = "pklfile"
os.makedirs(PKL_DIR, exist_ok=True)

class APIHealthResponse(BaseModel):
    message: str
    # request_id: str

class UploadResponse(BaseModel):
    message: str
    # request_id: str

class FaceRecognitionResponse(BaseModel):
    message: str
    # request_id: str

@app.get("/", response_model=APIHealthResponse,status_code=200)
async def api_health():
    return APIHealthResponse(message="API Health check", request_id=str(uuid.uuid4()))

@router.post("/add_user_id", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Endpoint to upload an image, extract faces and process the image.
    """
    request_id = str(uuid.uuid4())  
    img_stream = await file.read()
    nparr = np.frombuffer(img_stream, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    face_img = extract_faces_mtcnn(img_np)
    if face_img is None:
        raise HTTPException(status_code=400, detail="No face detected")
    
    try:
        filenames = pickle.load(open(os.path.join(PKL_DIR, 'FinalFilenames.pkl'), 'rb'))
    except (FileNotFoundError, EOFError):
        filenames = []
    
    try:
        existing_features = pickle.load(open(os.path.join(PKL_DIR, 'FeatureEmbeddings.pkl'), 'rb'))
        if not isinstance(existing_features, np.ndarray):
            existing_features = np.array(existing_features)
    except (FileNotFoundError, EOFError):
        existing_features = np.empty((0, 2048))
    
    # Extract only the filename (without extension)
    filename_without_ext = file.filename.split('.')[0]

    # Check if filename already exists
    if filename_without_ext in filenames:
        raise HTTPException(status_code=200, detail="File already processed")

    filenames.append(filename_without_ext)

    new_feature = feature_extractor(face_img).reshape(1, -1)
    
    if existing_features.size == 0:
        updated_features = new_feature
    else:
        updated_features = np.concatenate((existing_features, new_feature), axis=0)
    
    pickle.dump(filenames, open(os.path.join(PKL_DIR, 'FinalFilenames.pkl'), 'wb'))
    pickle.dump(updated_features, open(os.path.join(PKL_DIR, 'FeatureEmbeddings.pkl'), 'wb'))
    
    return UploadResponse(message="Image processed successfully", request_id=request_id)

@router.post("/verify")
async def upload_file(file: UploadFile = File(...)):
    try:
        request_id = str(uuid.uuid4())
        img_stream = await file.read()
        nparr = np.frombuffer(img_stream, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result_faces = faces_mtcnn(img_np)

        if result_faces['status'] == 'Empty':
            result_faces = [result_faces]
            return JSONResponse(content=result_faces, status_code=200)
        else:
            # Process each detected face
            results = []
            for face_image in result_faces['faces']:
                result = process_image(face_image)
                results.append(result)
            return JSONResponse(content=results, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.delete("/delete_id/{filename}")
async def delete_face_data(filename: str):
    """
    Delete a specific filename and its corresponding feature vector from the pickle files.
    """
    request_id = str(uuid.uuid4())
    filenames_path = os.path.join(PKL_DIR, 'FinalFilenames.pkl')
    features_path = os.path.join(PKL_DIR, 'FeatureEmbeddings.pkl')

    # Load filenames
    try:
        with open(filenames_path, 'rb') as f:
            filenames = pickle.load(f)
    except (FileNotFoundError, EOFError):
        return JSONResponse(
            content={"error": "Database not found."},
            status_code=404
        )

    # Load features
    try:
        with open(features_path, 'rb') as f:
            existing_features = pickle.load(f)
            if not isinstance(existing_features, np.ndarray):
                existing_features = np.array(existing_features)
    except (FileNotFoundError, EOFError):
        return JSONResponse(
            content={"error": "Database not found."},
            status_code=404
        )

    # Check if filename exists
    if filename not in filenames:
        return JSONResponse(
            content={"error": "User ID not found in database."},
            status_code=404
        )

    # Get index of filename
    index = filenames.index(filename)

    # Remove the filename and corresponding feature vector
    filenames.pop(index)
    updated_features = np.delete(existing_features, index, axis=0)

    # Save updated files
    with open(filenames_path, 'wb') as f:
        pickle.dump(filenames, f)

    with open(features_path, 'wb') as f:
        pickle.dump(updated_features, f)

    return JSONResponse(
        content={"message": "User ID deleted successfully", "filename": filename},
        status_code=200
    )

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
