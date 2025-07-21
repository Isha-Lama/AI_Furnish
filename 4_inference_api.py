from flask import Flask, request, jsonify, render_template
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Global variables for model and preprocessing objects
model = None
preprocessing_pipeline = None

# --- IMPORTANT: Ensure these lists match EXACTLY what's in 1_data_preparation.py ---
# These are used to create the semantic features if they are not provided by the user.
# The user's form has checkboxes for these, but this fallback ensures consistency
# if they are used as derived features during training.
seating_categories = ['chair', 'sofa', 'bench', 'couch', 'office_chair', 'bunk_bed']
table_categories = ['table', 'coffee_table', 'desk', 'dining_table', 'tv_stand']
storage_categories = ['cabinet', 'wardrobe', 'bookshelf', 'refrigerator']

def load_assets():
    global model, preprocessing_pipeline

    output_dir = 'output'
    model_path = os.path.join(output_dir, 'model.xgb')
    preprocessing_path = os.path.join(output_dir, 'preprocessing.pkl')

    if not os.path.exists(model_path) or not os.path.exists(preprocessing_path):
        logging.error("Model or preprocessing pipeline not found. Please run 1_data_preparation.py first.")
        # In a production app, you might want to gracefully shut down or return an error page.
        exit("Required model files are missing.")

    try:
        model = xgb.Booster()
        model.load_model(model_path)
        logging.info(f"XGBoost model loaded from {model_path}")

        with open(preprocessing_path, 'rb') as f:
            preprocessing_pipeline = pickle.load(f)
        logging.info(f"Preprocessing pipeline loaded from {preprocessing_path}")

    except Exception as e:
        logging.error(f"Error loading assets: {e}")
        exit("Failed to load model assets.")

@app.route('/')
def index():
    # You can pass the unique values for dropdowns from your encoders here
    # This makes the form dynamic, but requires the preprocessing_pipeline to be loaded.
    if preprocessing_pipeline:
        room_types = sorted(preprocessing_pipeline['room_type_encoder'].classes_.tolist())
        colors = sorted(preprocessing_pipeline['color_encoder'].classes_.tolist())
        materials = sorted(preprocessing_pipeline['material_encoder'].classes_.tolist())
    else:
        # Fallback to hardcoded if preprocessing_pipeline isn't loaded (e.g., during testing before full load)
        room_types = ['balcony', 'bathroom', 'bedroom', 'classroom', 'diningroom', 'guestroom', 'hallway', 'kidsroom', 'kitchen', 'livingroom', 'office', 'studyroom']
        colors = ['beige', 'black', 'blue', 'brown', 'gold', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'silver', 'turquoise', 'white', 'yellow']
        materials = ['bamboo', 'ceramic', 'concrete', 'fabric', 'glass', 'leather', 'metal', 'plastic', 'stone', 'wood']

    return render_template('index.html', room_types=room_types, colors=colors, materials=materials)


def apply_domain_logic(room_type, is_seating, is_table, is_storage, all_predictions_with_probs, category_label_encoder, num_top_predictions=3):
    """
    Applies domain-specific rules to adjust or re-rank predictions.
    
    Args:
        room_type (str): The room type from user input.
        is_seating (int): 1 if seating checkbox is checked, 0 otherwise.
        is_table (int): 1 if table checkbox is checked, 0 otherwise.
        is_storage (int): 1 if storage checkbox is checked, 0 otherwise.
        all_predictions_with_probs (list of dict): List of {'category': name, 'probability': prob} for all categories,
                                                     sorted by probability descending.
        category_label_encoder: The LabelEncoder for categories, used for inverse_transform.
        num_top_predictions (int): The number of top predictions to return.

    Returns:
        list of dict: A refined list of top predictions.
    """
    refined_predictions = []
    
    # Create a dictionary for quick lookup of current probabilities
    prob_dict = {pred['category']: float(pred['probability']) for pred in all_predictions_with_probs}

    # Initialize a list to hold final categories and their adjusted scores
    # We'll use the original probabilities as a base and modify them
    adjusted_scores = {}
    for cat_name, prob in prob_dict.items():
        adjusted_scores[cat_name] = prob

    # --- Rule 1: Room-specific adjustments ---
    if room_type == 'bathroom':
        # Promote bathroom-specific items, demote irrelevant ones
        bathroom_items = ['sink', 'bathtub', 'toilet', 'mirror', 'shower']
        demote_items = ['bookshelf', 'bed', 'sofa', 'dining_table']
        for item in bathroom_items:
            if item in adjusted_scores:
                adjusted_scores[item] *= 1.5  # Boost relevant items
        for item in demote_items:
            if item in adjusted_scores:
                adjusted_scores[item] *= 0.1 # Heavily penalize irrelevant items

    elif room_type == 'livingroom':
        livingroom_items = ['couch', 'sofa', 'tv_stand', 'coffee_table', 'armchair']
        demote_items = ['toilet', 'sink', 'bunk_bed', 'refrigerator']
        for item in livingroom_items:
            if item in adjusted_scores:
                adjusted_scores[item] *= 1.5
        for item in demote_items:
            if item in adjusted_scores:
                adjusted_scores[item] *= 0.1

    elif room_type == 'bedroom':
        bedroom_items = ['bed', 'wardrobe', 'nightstand', 'desk', 'chair']
        demote_items = ['toilet', 'sink', 'refrigerator', 'tv_stand']
        for item in bedroom_items:
            if item in adjusted_scores:
                adjusted_scores[item] *= 1.5
        for item in demote_items:
            if item in adjusted_scores:
                adjusted_scores[item] *= 0.1

    elif room_type == 'kitchen':
        kitchen_items = ['refrigerator', 'cabinet', 'dining_table', 'chair', 'stove'] # Assuming stove is a category
        demote_items = ['bed', 'sofa', 'bathtub', 'bookshelf']
        for item in kitchen_items:
            if item in adjusted_scores:
                adjusted_scores[item] *= 1.5
        for item in demote_items:
            if item in adjusted_scores:
                adjusted_scores[item] *= 0.1
                
    elif room_type == 'diningroom':
        diningroom_items = ['dining_table', 'chair', 'cabinet']
        demote_items = ['bed', 'sofa', 'toilet', 'tv_stand']
        for item in diningroom_items:
            if item in adjusted_scores:
                adjusted_scores[item] *= 1.5
        for item in demote_items:
            if item in adjusted_scores:
                adjusted_scores[item] *= 0.1

    elif room_type == 'office' or room_type == 'studyroom':
        office_items = ['desk', 'office_chair', 'bookshelf', 'cabinet']
        demote_items = ['bed', 'bathtub', 'sofa', 'refrigerator']
        for item in office_items:
            if item in adjusted_scores:
                adjusted_scores[item] *= 1.5
        for item in demote_items:
            if item in adjusted_scores:
                adjusted_scores[item] *= 0.1

    # --- Rule 2: Semantic feature adjustments (is_seating, is_table, is_storage) ---
    if is_seating == 1:
        # Prioritize seating items
        for cat in seating_categories:
            if cat in adjusted_scores:
                adjusted_scores[cat] *= 2.0  # Strong boost
        # Demote non-seating if a seating item is explicitly requested
        for cat_name in adjusted_scores:
            if cat_name not in seating_categories:
                adjusted_scores[cat_name] *= 0.5 # Slight penalty

    if is_table == 1:
        for cat in table_categories:
            if cat in adjusted_scores:
                adjusted_scores[cat] *= 2.0
        for cat_name in adjusted_scores:
            if cat_name not in table_categories:
                adjusted_scores[cat_name] *= 0.5

    if is_storage == 1:
        for cat in storage_categories:
            if cat in adjusted_scores:
                adjusted_scores[cat] *= 2.0
        for cat_name in adjusted_scores:
            if cat_name not in storage_categories:
                adjusted_scores[cat_name] *= 0.5

    # Combine all adjusted scores and re-sort
    sorted_adjusted_predictions = sorted(adjusted_scores.items(), key=lambda item: item[1], reverse=True)

    # Populate refined_predictions with top N, ensuring probabilities are formatted
    for i in range(min(num_top_predictions, len(sorted_adjusted_predictions))):
        category_name, adjusted_prob = sorted_adjusted_predictions[i]
        refined_predictions.append({
            "category": str(category_name),
            "probability": f"{adjusted_prob:.4f}"
        })
    
    return refined_predictions


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessing_pipeline is None:
        return jsonify({"error": "Model not loaded. Server is not ready."}), 500

    try:
        input_data = request.get_json()
        logging.info(f"Received prediction request: {input_data}")

        # Extract values with sensible defaults
        # Ensure consistent case with training data (lowercase)
        room_type = str(input_data.get('room_type', 'livingroom')).lower()
        color = str(input_data.get('color', 'white')).lower()
        material = str(input_data.get('material', 'wood')).lower()
        
        # Numerical features with defaults
        scale_x = float(input_data.get('scale_x', 1.0))
        scale_y = float(input_data.get('scale_y', 1.0))
        scale_z = float(input_data.get('scale_z', 1.0))
        rotation_y = float(input_data.get('rotation_y', 0.0))
        x_pos = float(input_data.get('x', 2.5))
        y_pos = float(input_data.get('y', 0.0))
        z_pos = float(input_data.get('z', 2.5))

        # --- IMPORTANT: Handle new semantic features (is_seating, is_table, is_storage) ---
        # These are now OPTIONAL inputs from the frontend. If not provided, default to 0.
        is_seating_input = int(input_data.get('is_seating', 0)) # Default to 0 if checkbox not checked
        is_table_input = int(input_data.get('is_table', 0))     # Default to 0 if checkbox not checked
        is_storage_input = int(input_data.get('is_storage', 0)) # Default to 0 if checkbox not checked

        # Create a temporary DataFrame to apply feature engineering consistently
        # Use a single-row DataFrame for consistency with preprocessing steps
        temp_df = pd.DataFrame([{
            'room_type': room_type,
            'color': color,
            'material': material,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'scale_z': scale_z,
            'rotation_y': rotation_y,
            'x': x_pos,
            'y': y_pos,
            'z': z_pos,
            'is_seating': is_seating_input, # Use the user's input for these
            'is_table': is_table_input,
            'is_storage': is_storage_input
        }])

        # --- Feature Engineering (MUST match 1_data_preparation.py) ---
        temp_df['volume'] = temp_df['scale_x'] * temp_df['scale_y'] * temp_df['scale_z']
        temp_df['aspect_ratio_xz'] = temp_df['scale_x'] / (temp_df['scale_z'] + 1e-6)
        temp_df['aspect_ratio_xy'] = temp_df['scale_x'] / (temp_df['scale_y'] + 1e-6)
        temp_df['distance_to_center'] = np.sqrt((temp_df['x'] - 2.5)**2 + (temp_df['z'] - 2.5)**2)
        temp_df['is_wall_near'] = ((temp_df['x'] < 1) | (temp_df['x'] > 4) | (temp_df['z'] < 1) | (temp_df['z'] > 4)).astype(int)
        temp_df['is_corner'] = ((temp_df['x'] < 1.5) & (temp_df['z'] < 1.5)) | \
                               ((temp_df['x'] > 3.5) & (temp_df['z'] > 3.5)) | \
                               ((temp_df['x'] < 1.5) & (temp_df['z'] > 3.5)) | \
                               ((temp_df['x'] > 3.5) & (temp_df['z'] < 1.5))
        temp_df['is_corner'] = temp_df['is_corner'].astype(int)
        temp_df['wall_corner_interaction'] = temp_df['is_wall_near'] * temp_df['is_corner']

        # --- Room-specific features (MUST match 1_data_preparation.py) ---
        # Get the unique rooms the model was trained on from the encoder
        trained_room_types = preprocessing_pipeline['room_type_encoder'].classes_
        for room in trained_room_types:
            temp_df[f'is_{room}'] = (temp_df['room_type'] == room).astype(int)

        # --- Encoding and Scaling (MUST match 1_data_preparation.py) ---
        scaler = preprocessing_pipeline['scaler']
        category_label_encoder = preprocessing_pipeline['category_label_encoder']
        room_type_encoder = preprocessing_pipeline['room_type_encoder']
        color_encoder = preprocessing_pipeline['color_encoder']
        material_encoder = preprocessing_pipeline['material_encoder']
        
        numerical_features = preprocessing_pipeline['numerical_features']
        feature_names = preprocessing_pipeline['feature_names'] # The full list of feature names

        # Encode categorical features
        # Handle unseen categories gracefully by replacing with a default or treating as 'unknown'
        try:
            temp_df['room_type_enc'] = room_type_encoder.transform(temp_df['room_type'])
        except ValueError:
            logging.warning(f"Unseen room_type: {room_type}. Assigning default encoding (first class).")
            temp_df['room_type_enc'] = room_type_encoder.transform([room_type_encoder.classes_[0]])[0]

        try:
            temp_df['color_enc'] = color_encoder.transform(temp_df['color'])
        except ValueError:
            logging.warning(f"Unseen color: {color}. Assigning default encoding (first class).")
            temp_df['color_enc'] = color_encoder.transform([color_encoder.classes_[0]])[0]

        try:
            temp_df['material_enc'] = material_encoder.transform(temp_df['material'])
        except ValueError:
            logging.warning(f"Unseen material: {material}. Assigning default encoding (first class).")
            temp_df['material_enc'] = material_encoder.transform([material_encoder.classes_[0]])[0]

        # Select and scale numerical features
        for col in numerical_features:
            if col not in temp_df.columns:
                logging.warning(f"Feature '{col}' not found in temp_df before scaling. Adding with default 0.")
                temp_df[col] = 0

        # Apply scaler to the selected numerical features
        # Ensure that only the columns listed in numerical_features are passed to the scaler
        df_for_scaling = temp_df[numerical_features].copy() # Use .copy() to avoid SettingWithCopyWarning
        scaled_numerical_features_df = pd.DataFrame(scaler.transform(df_for_scaling), columns=numerical_features)

        # Combine scaled numerical features with encoded categorical features
        # We need to reconstruct the full feature set in the correct order as expected by the model
        final_features_df = scaled_numerical_features_df.copy() # Start with scaled numericals
        final_features_df['room_type_enc'] = temp_df['room_type_enc'].iloc[0]
        final_features_df['color_enc'] = temp_df['color_enc'].iloc[0]
        final_features_df['material_enc'] = temp_df['material_enc'].iloc[0]
        
        # Ensure all features in feature_names are present and in the correct order
        # This is crucial for XGBoost with enable_categorical=True and feature_names
        X_predict = final_features_df[feature_names].values # This ensures correct order

        # Create DMatrix for prediction
        dpredict = xgb.DMatrix(X_predict, enable_categorical=True, feature_names=feature_names)

        # Make prediction
        pred_proba = model.predict(dpredict)
        
        # Get all predictions and their category names
        all_predictions_with_probs = []
        for idx, prob in enumerate(pred_proba[0]):
            category_name = category_label_encoder.inverse_transform([idx])[0]
            all_predictions_with_probs.append({
                "category": str(category_name),
                "probability": prob
            })
        
        # Sort by probability in descending order to prepare for domain logic
        all_predictions_with_probs.sort(key=lambda x: x['probability'], reverse=True)

        # Apply domain logic to refine the top predictions
        top_predictions_refined = apply_domain_logic(
            room_type=room_type, 
            is_seating=is_seating_input, 
            is_table=is_table_input, 
            is_storage=is_storage_input,
            all_predictions_with_probs=all_predictions_with_probs,
            category_label_encoder=category_label_encoder,
            num_top_predictions=3 # Get top 3 refined predictions
        )

        # The predicted_category (single best prediction) will be the first in the refined list
        predicted_category = top_predictions_refined[0]['category'] if top_predictions_refined else "No prediction"

        response = {
            "predicted_category": str(predicted_category),
            "top_predictions": top_predictions_refined
        }
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

# Run this once when the Flask app starts
with app.app_context():
    load_assets()

if __name__ == '__main__':
    # When running locally for development, set debug=True
    # In production, use a production-ready WSGI server like Gunicorn.
    app.run(debug=True, host='0.0.0.0', port=5000)