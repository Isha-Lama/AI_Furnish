import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- GLOBAL HOLDERS ---
model = None
preprocessing_pipeline = None

# --- Load model and pipeline ---
def load_assets():
    global model, preprocessing_pipeline
    output_dir = 'output'
    model_path = os.path.join(output_dir, 'model.xgb')
    preprocessing_path = os.path.join(output_dir, 'preprocessing.pkl')

    if not os.path.exists(model_path) or not os.path.exists(preprocessing_path):
        logging.error("Model or preprocessing pipeline not found. Please run 1_data_preparation.py first.")
        exit("Required model files are missing.")

    try:
        model = xgb.Booster()
        model.load_model(model_path)
        logging.info(f"Model loaded from {model_path}")

        with open(preprocessing_path, 'rb') as f:
            preprocessing_pipeline = pickle.load(f)
        logging.info(f"Preprocessing pipeline loaded from {preprocessing_path}")

    except Exception as e:
        logging.error(f"Error loading assets: {e}")
        exit("Failed to load model assets.")

def contextual_prioritization(pred_proba, room_type, is_seating, is_table, is_storage, category_label_encoder):
    label_list = category_label_encoder.classes_.tolist()
    boost = np.zeros_like(pred_proba)

    if room_type == 'bathroom':
        for item in ['sink', 'bathtub']:
            if item in label_list:
                boost[label_list.index(item)] += 0.2

    elif room_type == 'livingroom':
        if is_seating:
            for item in ['couch', 'sofa', 'chair']:
                if item in label_list:
                    boost[label_list.index(item)] += 0.2
        if is_table:
            for item in ['coffee_table', 'table']:
                if item in label_list:
                    boost[label_list.index(item)] += 0.1
        for item in ['tv_stand', 'lamp']:
            if item in label_list:
                boost[label_list.index(item)] += 0.05

    elif room_type == 'bedroom':
        for item in ['bed', 'wardrobe', 'lamp']:
            if item in label_list:
                boost[label_list.index(item)] += 0.1
        if is_seating:
            for item in ['chair', 'bench']:
                if item in label_list:
                    boost[label_list.index(item)] += 0.1

    elif room_type == 'kitchen':
        for item in ['sink', 'refrigerator', 'cabinet']:
            if item in label_list:
                boost[label_list.index(item)] += 0.1

    elif room_type == 'office':
        if is_seating:
            for item in ['office_chair', 'chair']:
                if item in label_list:
                    boost[label_list.index(item)] += 0.15
        if is_table:
            for item in ['desk', 'table']:
                if item in label_list:
                    boost[label_list.index(item)] += 0.1

    pred_proba_adjusted = pred_proba + boost
    pred_proba_adjusted /= pred_proba_adjusted.sum()
    return pred_proba_adjusted

def predict_category(input_data):
    global model, preprocessing_pipeline

    df = pd.DataFrame([input_data])
    scaler = preprocessing_pipeline['scaler']
    category_encoder = preprocessing_pipeline['category_label_encoder']
    room_encoder = preprocessing_pipeline['room_type_encoder']
    color_encoder = preprocessing_pipeline['color_encoder']
    material_encoder = preprocessing_pipeline['material_encoder']
    numerical_features = preprocessing_pipeline['numerical_features']
    feature_names = preprocessing_pipeline['feature_names']

    df['volume'] = df['scale_x'] * df['scale_y'] * df['scale_z']
    df['aspect_ratio_xz'] = df['scale_x'] / (df['scale_z'] + 1e-6)
    df['aspect_ratio_xy'] = df['scale_x'] / (df['scale_y'] + 1e-6)
    df['distance_to_center'] = np.sqrt((df['x'] - 2.5)**2 + (df['z'] - 2.5)**2)
    df['is_wall_near'] = ((df['x'] < 1) | (df['x'] > 4) | (df['z'] < 1) | (df['z'] > 4)).astype(int)
    df['is_corner'] = (
        ((df['x'] < 1.5) & (df['z'] < 1.5)) |
        ((df['x'] > 3.5) & (df['z'] > 3.5)) |
        ((df['x'] < 1.5) & (df['z'] > 3.5)) |
        ((df['x'] > 3.5) & (df['z'] < 1.5))
    ).astype(int)
    df['wall_corner_interaction'] = df['is_wall_near'] * df['is_corner']

    for room in room_encoder.classes_:
        df[f'is_{room}'] = (df['room_type'] == room).astype(int)

    try:
        df['room_type_enc'] = room_encoder.transform(df['room_type'])
    except:
        df['room_type_enc'] = room_encoder.transform([room_encoder.classes_[0]])[0]

    try:
        df['color_enc'] = color_encoder.transform(df['color'])
    except:
        df['color_enc'] = color_encoder.transform([color_encoder.classes_[0]])[0]

    try:
        df['material_enc'] = material_encoder.transform(df['material'])
    except:
        df['material_enc'] = material_encoder.transform([material_encoder.classes_[0]])[0]

    for col in numerical_features:
        if col not in df.columns:
            df[col] = 0

    scaled = scaler.transform(df[numerical_features])
    feature_dict = {col: df[col].iloc[0] for col in numerical_features}
    feature_dict['room_type_enc'] = df['room_type_enc'].iloc[0]
    feature_dict['color_enc'] = df['color_enc'].iloc[0]
    feature_dict['material_enc'] = df['material_enc'].iloc[0]
    ordered = [feature_dict[name] for name in feature_names]
    X = np.array([ordered])
    dmatrix = xgb.DMatrix(X, enable_categorical=True, feature_names=feature_names)

    raw_pred = model.predict(dmatrix)[0]
    adjusted_pred = contextual_prioritization(
        raw_pred, df['room_type'].iloc[0],
        df['is_seating'].iloc[0],
        df['is_table'].iloc[0],
        df['is_storage'].iloc[0],
        category_encoder
    )

    top_idxs = np.argsort(adjusted_pred)[::-1][:3]
    top_cats = category_encoder.inverse_transform(top_idxs)
    top_probs = adjusted_pred[top_idxs]

    return {
        "predicted_category": top_cats[0],
        "top_predictions": [
            {"category": cat, "probability": f"{prob:.4f}"}
            for cat, prob in zip(top_cats, top_probs)
        ]
    }
