# import numpy as np
# from PIL import Image
# from keras.preprocessing import image
# from keras_vggface.vggface import VGGFace
# from keras_vggface.utils import preprocess_input
# from mtcnn import MTCNN
# from io import BytesIO

# def feature_extractor(img):
#     """
#     Extract features from an image using VGGFace model.
#     Args:
#     - img: A PIL Image object.
    
#     Returns:
#     - A flattened numpy array representing the features of the image.
#     """
#     model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
#     img = img.resize((224, 224))
#     img_array = image.img_to_array(img)
#     expanded_img = np.expand_dims(img_array, axis=0)
#     preprocessed_img = preprocess_input(expanded_img)
#     result = model.predict(preprocessed_img).flatten()
#     return result

# def extract_faces_mtcnn(image_bytes, required_size=(224, 224)):
#     """
#     Extract faces from an image using MTCNN detector.
#     Args:
#     - image_bytes: The byte data of the image.
#     - required_size: The target size for resizing the detected face.
    
#     Returns:
#     - A PIL Image object containing the extracted face, or None if no face is detected.
#     """
#     detector = MTCNN()
#     img = Image.open(BytesIO(image_bytes)).convert("RGB")
#     img_array = np.array(img)
#     faces = detector.detect_faces(img_array)
    
#     if not faces:
#         return None
    
#     for face_info in faces:
#         x, y, w, h = face_info['box']
#         x, y = max(x, 0), max(y, 0)
#         face = img_array[y:y + h, x:x + w]
#         image_obj = Image.fromarray(face)
#         image_obj = image_obj.resize(required_size)
#         return image_obj






import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from mtcnn import MTCNN
from io import BytesIO
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
detector = MTCNN()

def feature_extractor(img_data):
    img_array = image.img_to_array(img_data).astype('float32')
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def extract_faces_mtcnn(img_data, required_size=(224, 224)):
    img = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(img)
    converted_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
    faces = detector.detect_faces(converted_image)
    if not faces:
        return None

    face_images = []
    for face_info in faces:
        x, y, w, h = face_info['box']
        x, y = max(x, 0), max(y, 0)
        face = img_data[y:y + h, x:x + w]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        # face_array = np.asarray(image)
        # face_array = cv2.bilateralFilter(face_array, 9, 75, 75)
        return image

def faces_mtcnn(img_data, required_size=(224, 224)):
    img = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(img)
    converted_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
    faces = detector.detect_faces(converted_image)
    if not faces:
        return {'status': 'Empty', 'User ID': "Face not Found"}

    face_images = []
    for face_info in faces:
        x, y, w, h = face_info['box']
        x, y = max(x, 0), max(y, 0)
        face = img_data[y:y + h, x:x + w]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        face_array = cv2.bilateralFilter(face_array, 9, 75, 75)
        face_images.append(face_array)

    return {'status': 'Success', 'faces': face_images}

def process_image(img_data):

    PKL_DIR = "pklfile"
    # Define paths for the pickle files
    filenames_path = os.path.join(PKL_DIR, 'FinalFilenames.pkl')
    feature_list_path = os.path.join(PKL_DIR, 'FeatureEmbeddings.pkl')

    # Check if the pickle files exist
    if not os.path.exists(filenames_path) or not os.path.exists(feature_list_path):
        filenames = None
        feature_list = None
    else:
        # Load the feature list and filenames from files (Pickle)
        filenames = np.array(pickle.load(open(filenames_path, 'rb')))
        feature_list = pickle.load(open(feature_list_path, 'rb'))

    # If the pickle files are not loaded (i.e., the database is empty)
    if filenames is None or feature_list is None:
        return {'status': 'Success', 'User ID': "Your Database is Empty"}


    result = feature_extractor(img_data)
    similarity = cosine_similarity(result.reshape(1, -1), feature_list)
    max_similarity = np.max(similarity)
    
    max_similarity_percentage = max_similarity * 100

    if max_similarity >= 0.60:
        index = np.argmax(similarity)
        tracking_number = f"{filenames[index]} ({max_similarity_percentage:.2f}%)"
    else:
        tracking_number = f"Unknown ({max_similarity_percentage:.2f}%)"
    
    return {'status': 'Success', 'User ID': tracking_number}