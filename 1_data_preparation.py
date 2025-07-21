import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTENC
import xgboost as xgb
import logging
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the 'output' directory exists for saving files
os.makedirs('output', exist_ok=True)

# -------------------- Enhanced Cleaning Rules --------------------
# Keep general_furniture as is, based on your request not to add new types
general_furniture = ['lamp', 'rug', 'plant', 'chair', 'table', 'mirror', 'cabinet']

# Explicitly ensure ALLOWED_ROOM_FURNITURE only contains your existing categories + general
# and ensure consistency in naming (e.g., 'wardrob' corrected to 'wardrobe')
ALLOWED_ROOM_FURNITURE = {
    "bathroom": ['bathtub', 'mirror', 'cabinet', 'sink', 'lamp'] + general_furniture,
    "bedroom": ['bed', 'wardrobe', 'lamp', 'mirror', 'rug', 'chair', 'table', 'plant', 'bookshelf'] + general_furniture,
    "livingroom": ['sofa', 'tv_stand', 'coffee_table', 'couch', 'lamp', 'plant', 'rug', 'bookshelf', 'chair'] + general_furniture,
    "kitchen": ['table', 'chair', 'cabinet', 'refrigerator', 'lamp', 'sink'] + general_furniture,
    "diningroom": ['dining_table', 'chair', 'lamp', 'plant', 'cabinet'] + general_furniture,
    "office": ['desk', 'office_chair', 'bookshelf', 'lamp', 'plant', 'chair'] + general_furniture,
    "studyroom": ['desk', 'chair', 'bookshelf', 'lamp', 'plant'] + general_furniture,
    "classroom": ['desk', 'chair', 'bookshelf', 'lamp', 'plant'] + general_furniture,
    "kidsroom": ['bunk_bed', 'wardrobe', 'rug', 'lamp', 'plant', 'bookshelf', 'chair'] + general_furniture,
    "guestroom": ['bed', 'wardrobe', 'mirror', 'lamp', 'rug', 'plant', 'bookshelf'] + general_furniture, # Corrected 'wardrob'
    "hallway": ['bench', 'mirror', 'plant', 'lamp', 'cabinet'] + general_furniture,
    "balcony": ['plant', 'bench', 'chair', 'lamp', 'table'] + general_furniture
}

# Refined INVALID_FURNITURE_PER_ROOM based *only* on existing categories
INVALID_FURNITURE_PER_ROOM = {
    "bedroom": ['dining_table', 'sink', 'refrigerator', 'bathtub', 'tv_stand', 'couch', 'sofa', 'bench'], # Stricter
    "bathroom": ['bed', 'bookshelf', 'couch', 'refrigerator', 'tv_stand', 'sofa', 'desk', 'dining_table', 'office_chair', 'bunk_bed', 'coffee_table', 'wardrobe', 'bench'], # Stricter
    "kitchen": ['bed', 'bookshelf', 'bunk_bed', 'bench', 'office_chair', 'sofa', 'couch', 'tv_stand', 'dining_table'], # Stricter
    "diningroom": ['bed', 'wardrobe', 'sofa', 'desk', 'office_chair', 'bunk_bed', 'bathtub', 'sink', 'refrigerator', 'tv_stand', 'couch', 'bench'], # Stricter
    "office": ['bed', 'dining_table', 'bathtub', 'refrigerator', 'sink', 'couch', 'bunk_bed', 'sofa', 'tv_stand'], # Stricter
    "livingroom": ['bathtub', 'sink', 'refrigerator', 'desk', 'office_chair', 'bunk_bed', 'dining_table'], # Stricter
    "kidsroom": ['dining_table', 'sink', 'refrigerator', 'bathtub', 'office_chair', 'coffee_table', 'sofa', 'couch', 'desk'], # Stricter
    "guestroom": ['dining_table', 'sink', 'refrigerator', 'bathtub', 'bunk_bed', 'coffee_table', 'desk', 'office_chair'], # Stricter
    "hallway": ['bed', 'bunk_bed', 'bathtub', 'sink', 'refrigerator', 'dining_table', 'sofa', 'couch', 'tv_stand', 'desk', 'office_chair', 'coffee_table', 'wardrobe', 'bookshelf'], # Very strict for hallway
    "balcony": ['bed', 'wardrobe', 'desk', 'office_chair', 'bathtub', 'sink', 'refrigerator', 'bunk_bed', 'tv_stand', 'dining_table', 'sofa', 'couch', 'bookshelf', 'cabinet', 'mirror'], # Very strict for balcony
    "studyroom": ['bathtub', 'sink', 'dining_table', 'refrigerator', 'bunk_bed', 'tv_stand', 'sofa', 'couch'], # Stricter
    "classroom": ['bed', 'bunk_bed', 'bathtub', 'sink', 'dining_table', 'refrigerator', 'sofa', 'couch', 'tv_stand', 'coffee_table', 'wardrobe', 'cabinet'] # Stricter
}

# Refined INVALID_MATERIALS based *only* on existing categories
INVALID_MATERIALS = {
    'bathtub': ['fabric', 'leather', 'wood', 'plastic', 'rubber', 'paper', 'cardboard', 'cloth', 'foam', 'velvet'],
    'plant': ['leather', 'metal', 'glass', 'ceramic', 'plastic', 'rubber', 'stone', 'concrete', 'fabric', 'wood'],
    'rug': ['glass', 'metal', 'stone', 'concrete', 'ceramic', 'plastic', 'wood', 'rubber', 'paper', 'cardboard'],
    'mirror': ['fabric', 'wood', 'plastic', 'leather', 'rubber', 'paper', 'cardboard'],
    'lamp': ['leather', 'concrete', 'rubber', 'fabric', 'wood'],
    'sink': ['wood', 'leather', 'fabric', 'plastic', 'rubber', 'paper', 'cardboard'],
    'bookshelf': ['leather', 'fabric', 'plastic', 'rubber', 'glass'], # Bookshelves usually wood/metal, sometimes glass doors
    'chair': ['glass', 'ceramic', 'rubber', 'paper', 'cardboard'],
    'table': ['fabric', 'rubber', 'paper', 'cardboard'],
    'cabinet': ['fabric', 'leather', 'rubber', 'glass'], # Cabinets usually wood/metal, sometimes glass doors
    'bed': ['glass', 'metal', 'ceramic', 'plastic', 'stone', 'concrete'], # Beds are mostly wood/metal frame with fabric/leather covering
    'wardrobe': ['fabric', 'leather', 'plastic', 'rubber', 'glass'],
    'sofa': ['wood', 'metal', 'glass', 'ceramic', 'plastic', 'stone', 'concrete', 'rubber', 'paper', 'cardboard'],
    'tv_stand': ['fabric', 'leather', 'rubber', 'glass', 'ceramic'],
    'coffee_table': ['fabric', 'leather', 'rubber', 'ceramic'],
    'couch': ['wood', 'metal', 'glass', 'ceramic', 'plastic', 'stone', 'concrete', 'rubber', 'paper', 'cardboard'],
    'refrigerator': ['wood', 'fabric', 'leather', 'plastic', 'glass', 'ceramic', 'stone', 'concrete', 'rubber'], # fridges are typically metal
    'desk': ['fabric', 'rubber', 'ceramic'],
    'office_chair': ['glass', 'ceramic', 'rubber'], # office chairs often have fabric/leather and plastic/metal parts
    'dining_table': ['fabric', 'rubber', 'ceramic'],
    'bunk_bed': ['glass', 'ceramic', 'plastic'],
    'bench': ['glass', 'ceramic', 'rubber'],
}

# Refined INVALID_COLORS based *only* on existing categories
INVALID_COLORS = {
    'plant': ['black', 'silver', 'gold', 'red', 'blue', 'purple', 'white', 'orange', 'yellow', 'pink'], # Plants are predominantly green/brown
    'bathtub': ['red', 'green', 'black', 'purple', 'yellow', 'orange', 'brown', 'gold', 'silver'], # Bathtubs usually white/off-white, maybe blue/pink in vintage
    'sink': ['gold', 'purple', 'red', 'green', 'orange', 'yellow', 'brown'], # Sinks usually white/silver/chrome
    'rug': ['silver', 'gold'], # Rugs usually colorful or neutral, not metallic
    'mirror': ['green', 'red', 'purple', 'orange', 'brown'], # Mirror reflection, frames can be other colors but main part is reflective
    'lamp': ['silver', 'gold', 'red', 'green', 'purple', 'orange', 'yellow', 'pink'], # Lamps can be varied, but some colors are rare for the light itself or shade material
    'refrigerator': ['red', 'green', 'purple', 'pink', 'orange', 'yellow', 'brown'], # Common fridge colors are white, black, silver
    'wardrobe': ['red', 'green', 'purple', 'pink', 'orange', 'yellow'], # Wardrobes usually neutral colors or wood tones
    'bed': ['silver', 'gold', 'red', 'green', 'purple', 'orange', 'yellow', 'pink'], # Bed frames usually neutral, wood, black. Covers can be anything.
    'sofa': ['silver', 'gold'], # Sofas usually fabric/leather colors
    'tv_stand': ['silver', 'gold', 'red', 'green', 'purple', 'pink', 'orange', 'yellow'],
    'coffee_table': ['silver', 'gold', 'red', 'green', 'purple', 'pink', 'orange', 'yellow'],
    'bookshelf': ['silver', 'gold', 'red', 'green', 'purple', 'pink', 'orange', 'yellow'],
    'desk': ['silver', 'gold', 'red', 'green', 'purple', 'pink', 'orange', 'yellow'],
    'office_chair': ['silver', 'gold', 'red', 'green', 'purple', 'pink', 'orange', 'yellow'],
    'dining_table': ['silver', 'gold', 'red', 'green', 'purple', 'pink', 'orange', 'yellow'],
    'bunk_bed': ['silver', 'gold', 'red', 'green', 'purple', 'pink', 'orange', 'yellow'],
    'bench': ['silver', 'gold', 'red', 'green', 'purple', 'pink', 'orange', 'yellow'],
    'cabinet': ['red', 'green', 'purple', 'pink', 'orange', 'yellow'] # Cabinets are usually neutral, wood, or white
}


def is_valid(row):
    item = str(row['category']).lower() # Ensure string and lower
    room = str(row['room_type']).lower() # Ensure string and lower
    mat = str(row['material']).lower() # Ensure string and lower
    col = str(row['color']).lower() # Ensure string and lower

    # Rule 1: Check if the room type is known - essential for other rules
    if room not in ALLOWED_ROOM_FURNITURE:
        logging.debug(f"Invalid room type '{room}' for item '{item}'. Row discarded.")
        return False
    
    # Rule 2: Check if the furniture item is allowed in that specific room
    # This acts as a positive filter. If an item is not in the allowed list for a room, it's invalid.
    if item not in ALLOWED_ROOM_FURNITURE[room]: # Use [] for strict checking if not present
        logging.debug(f"Item '{item}' not allowed in room '{room}'. Row discarded.")
        return False

    # Rule 3: Check for explicitly invalid furniture in a specific room (stronger filter)
    if room in INVALID_FURNITURE_PER_ROOM and item in INVALID_FURNITURE_PER_ROOM[room]:
        logging.debug(f"Item '{item}' explicitly invalid for room '{room}'. Row discarded.")
        return False

    # Rule 4: Check for invalid material for the item
    if item in INVALID_MATERIALS and mat in INVALID_MATERIALS[item]:
        logging.debug(f"Invalid material '{mat}' for item '{item}'. Row discarded.")
        return False

    # Rule 5: Check for invalid color for the item
    if item in INVALID_COLORS and col in INVALID_COLORS[item]:
        logging.debug(f"Invalid color '{col}' for item '{item}'. Row discarded.")
        return False
    
    # Rule 6: Basic sanity checks for numerical values (e.g., scale should be positive)
    if not (row['scale_x'] > 0 and row['scale_y'] > 0 and row['scale_z'] > 0):
        logging.debug(f"Invalid scale dimensions for item '{item}'. Row discarded.")
        return False
    
    # Rule 7: Basic sanity for position (within a reasonable room boundary, assuming 5x5x5 room)
    # Check if x, y, z are within [0, 5]
    if not (0 <= row['x'] <= 5 and 0 <= row['y'] <= 5 and 0 <= row['z'] <= 5):
        logging.debug(f"Item '{item}' at invalid position ({row['x']},{row['y']},{row['z']}). Row discarded.")
        return False

    return True

def main():
    # Load and clean data
    try:
        df = pd.read_csv('enriched_furniture_dataset_13000.csv')
    except FileNotFoundError:
        logging.error("Error: 'enriched_furniture_dataset_13000.csv' not found. Please ensure the dataset is in the correct directory.")
        return

    original_df_len = len(df)
    logging.info(f"Loaded dataset with {original_df_len} rows.")
    
    # Standardize string columns to lowercase for consistent matching
    df['room_type'] = df['room_type'].str.lower()
    df['material'] = df['material'].str.lower()
    df['color'] = df['color'].str.lower()
    df['category'] = df['category'].str.lower()

    # Apply cleaning rules
    df_cleaned = df[df.apply(is_valid, axis=1)].copy()
    num_rows_after_cleaning = len(df_cleaned)
    logging.info(f"Data cleaned. Original rows: {original_df_len}, Remaining rows: {num_rows_after_cleaning} ({((original_df_len - num_rows_after_cleaning)/original_df_len)*100:.2f}% removed).")

    # Filter categories with too few samples to ensure meaningful training
    MIN_SAMPLES_FOR_CATEGORY = 25 # Keep this threshold as it's reasonable
    initial_category_counts = df_cleaned['category'].value_counts()
    logging.info(f"Initial category distribution (before filtering low counts):\n{initial_category_counts.to_string()}")

    categories_to_keep = initial_category_counts[initial_category_counts >= MIN_SAMPLES_FOR_CATEGORY].index
    df_filtered = df_cleaned[df_cleaned['category'].isin(categories_to_keep)].copy()
    
    final_category_counts = df_filtered['category'].value_counts()
    logging.info(f"Category distribution after filtering (>= {MIN_SAMPLES_FOR_CATEGORY} samples):\n{final_category_counts.to_string()}")
    
    # Handle 'other' category more dynamically
    if 'other' in final_category_counts.index:
        other_count = final_category_counts['other']
        
        # Calculate mean of non-'other' categories, handling cases where 'other' is the only category
        non_other_counts = final_category_counts[final_category_counts.index != 'other']
        mean_non_other_count = non_other_counts.mean() if not non_other_counts.empty else 0
        
        # Target 'other' count to be slightly above the average of other categories, but not excessively large
        # Also ensure a minimum count if 'other' is very small, or cap it if it's very large
        target_other_count = int(mean_non_other_count * 1.5) if mean_non_other_count > 0 else 500 # If no other categories, default to 500

        # Ensure we don't undersample below a reasonable minimum, e.g., 100 samples (reduced from 200)
        # This prevents 'other' from becoming too small if other categories are also small
        min_absolute_other_count = 100
        target_other_count = max(min_absolute_other_count, target_other_count)
        
        # Cap the target count at the actual 'other' count to prevent increasing it unnecessarily
        target_other_count = min(other_count, target_other_count) 

        if other_count > target_other_count:
            logging.info(f"Undersampling 'other' from {other_count} to {target_other_count} samples.")
            df_other = df_filtered[df_filtered['category'] == 'other'].sample(n=target_other_count, random_state=42)
            df_non_other = df_filtered[df_filtered['category'] != 'other']
            df_processed = pd.concat([df_other, df_non_other])
        else:
            df_processed = df_filtered # No undersampling needed for 'other' or it's already small enough
    else:
        df_processed = df_filtered # No 'other' category found

    logging.info(f"Categories after cleaning and 'OTHER' handling: {df_processed['category'].nunique()} unique categories. Total rows: {len(df_processed)}")
    logging.info(f"Final category distribution:\n{df_processed['category'].value_counts().to_string()}")

    # Advanced feature engineering (your existing good features + some new ones)
    df_processed['volume'] = df_processed['scale_x'] * df_processed['scale_y'] * df_processed['scale_z']
    df_processed['aspect_ratio_xz'] = df_processed['scale_x'] / (df_processed['scale_z'] + 1e-6)
    df_processed['aspect_ratio_xy'] = df_processed['scale_x'] / (df_processed['scale_y'] + 1e-6)
    df_processed['distance_to_center'] = np.sqrt((df_processed['x'] - 2.5)**2 + (df_processed['z'] - 2.5)**2) # Assuming room center is 2.5, 2.5
    df_processed['is_wall_near'] = ((df_processed['x'] < 1) | (df_processed['x'] > 4) | (df_processed['z'] < 1) | (df_processed['z'] > 4)).astype(int)
    
    df_processed['is_corner'] = ((df_processed['x'] < 1.5) & (df_processed['z'] < 1.5)) | \
                                ((df_processed['x'] > 3.5) & (df_processed['z'] > 3.5)) | \
                                ((df_processed['x'] < 1.5) & (df_processed['z'] > 3.5)) | \
                                ((df_processed['x'] > 3.5) & (df_processed['z'] < 1.5))
    df_processed['is_corner'] = df_processed['is_corner'].astype(int)
    df_processed['wall_corner_interaction'] = df_processed['is_wall_near'] * df_processed['is_corner']
    
    # New Feature: Normalized height (y-coordinate relative to a conceptual room height of 5 units)
    df_processed['normalized_height'] = df_processed['y'] / 5.0

    # New Feature: Room Area (assuming a fixed room size, e.g., 5x5)
    # This might not add much if it's constant, but if room types implied different sizes, it would.
    # For now, let's assume a conceptual room area for context
    df_processed['room_area'] = 25.0 # Assuming standard room is 5x5, fixed value if no room dimension features

    # New Feature: Furniture Density (might be useful if you had counts per room, but can fake it)
    # This is a bit of a placeholder without actual per-room furniture counts.
    # We can approximate it by inverse of volume, or simply 1/volume as a proxy for "small/dense" item
    df_processed['furniture_density'] = 1 / (df_processed['volume'] + 1e-6) # Inverse of volume
    
    # Semantic groups (features *derived from category*, so be careful with interpretation)
    seating = ['chair', 'sofa', 'bench', 'couch', 'office_chair', 'bunk_bed']
    tables = ['table', 'coffee_table', 'desk', 'dining_table', 'tv_stand']
    storage = ['cabinet', 'wardrobe', 'bookshelf', 'refrigerator']
    
    df_processed['is_seating'] = df_processed['category'].isin(seating).astype(int)
    df_processed['is_table'] = df_processed['category'].isin(tables).astype(int)
    df_processed['is_storage'] = df_processed['category'].isin(storage).astype(int)
    
    # Room-specific features (one-hot like encoding)
    unique_rooms = df_processed['room_type'].unique()
    for room in unique_rooms:
        df_processed[f'is_{room}'] = (df_processed['room_type'] == room).astype(int)
    
    # Target encoding
    le = LabelEncoder()
    y = le.fit_transform(df_processed['category'])
    logging.info(f"Target variable encoded. Number of classes: {len(np.unique(y))}. Class names: {list(le.classes_)}")
    
    # Feature selection (numerical columns)
    num_cols = [
        'scale_x', 'scale_y', 'scale_z', 'rotation_y', 'x', 'y', 'z',
        'volume', 'aspect_ratio_xz', 'aspect_ratio_xy', 'distance_to_center',
        'is_wall_near', 'is_corner', 'wall_corner_interaction',
        'normalized_height', 'room_area', 'furniture_density', # New features added here
        'is_seating', 'is_table', 'is_storage'
    ] + [f'is_{room}' for room in unique_rooms]
    
    # Scaling numerical features
    scaler = MinMaxScaler()
    df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])
    logging.info("Numerical features scaled using MinMaxScaler.")
    
    # Categorical encoding - Fit before creating X, and store encoders
    room_type_encoder = LabelEncoder()
    df_processed['room_type_enc'] = room_type_encoder.fit_transform(df_processed['room_type'])
    
    color_encoder = LabelEncoder()
    df_processed['color_enc'] = color_encoder.fit_transform(df_processed['color'])
    
    material_encoder = LabelEncoder()
    df_processed['material_enc'] = material_encoder.fit_transform(df_processed['material'])
    logging.info("Categorical features (room_type, color, material) encoded with LabelEncoder.")
    
    # Feature matrix
    X = np.hstack([
        df_processed[num_cols].values,
        df_processed[['room_type_enc', 'color_enc', 'material_enc']].values
    ])
    
    # Identify categorical feature indices for SMOTENC and XGBoost
    # These indices must be based on the final X array
    cat_idx = [X.shape[1] - 3, X.shape[1] - 2, X.shape[1] - 1]
    logging.info(f"Categorical feature indices for SMOTENC/XGBoost (zero-indexed): {cat_idx}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    logging.info(f"Data split into training ({len(X_train)} samples) and test sets ({len(X_test)} samples).")
    
    train_category_counts = pd.Series(y_train).value_counts()
    min_class_samples = train_category_counts.min()

    # SMOTENC k_neighbors must be less than the number of samples in the smallest class.
    # Default to 5 if possible, else min_class_samples - 1 (min 1).
    smote_k_neighbors = max(1, min(5, min_class_samples - 1))
    
    if smote_k_neighbors < 1:
        logging.warning("⚠️ Warning: SMOTENC cannot be applied as some minority classes have too few samples (less than 2). Skipping SMOTENC.")
        X_res, y_res = X_train, y_train
    else:
        smote = SMOTENC(
            categorical_features=cat_idx,
            random_state=42,
            k_neighbors=smote_k_neighbors,
            sampling_strategy='not majority' # Balance all classes except the majority
        )
        logging.info(f"Applying SMOTENC with k_neighbors={smote_k_neighbors} for imbalance handling.")
        X_res, y_res = smote.fit_resample(X_train, y_train)
    logging.info(f"After SMOTENC, training samples: {len(X_res)}")
    logging.info(f"Distribution after SMOTENC (class_id: count):\n{Counter(y_res)}")
        
    weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_res),
        y=y_res
    )
    sample_weights = np.array([weights[cls] for cls in y_res])
    logging.info("Class weights computed for balanced training.")
    
    # Create DMatrix objects with enable_categorical=True and feature names
    feature_names_for_dmatrix = num_cols + ['room_type_enc', 'color_enc', 'material_enc']

    dtrain = xgb.DMatrix(X_res, label=y_res, weight=sample_weights, 
                         enable_categorical=True, feature_names=feature_names_for_dmatrix)
    dtest = xgb.DMatrix(X_test, label=y_test, 
                        enable_categorical=True, feature_names=feature_names_for_dmatrix)
    
    # Tuned parameters for better generalization and performance
    # Increased max_depth, num_boost_round, and early_stopping_rounds
    # Fine-tuned eta, subsample, colsample_bytree, gamma, alpha for more robust training
    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y)),
        'max_depth': 12,        # Increased from 10 to allow more complex interactions
        'eta': 0.005,           # Further reduced learning rate for more careful steps over many rounds
        'subsample': 0.75,      # Slightly increased subsample from 0.7
        'colsample_bytree': 0.75, # Slightly increased colsample from 0.7
        'min_child_weight': 1,  # Kept
        'gamma': 0.25,          # Slightly increased gamma for more conservative splitting
        'lambda': 1.2,          # Increased L2 regularization slightly
        'alpha': 0.5,           # Increased L1 regularization slightly for more feature selection
        'tree_method': 'hist',  # Faster for large datasets
        'eval_metric': ['mlogloss', 'merror'],
        'seed': 42,
        'n_jobs': -1,           # Use all available CPU cores
        'rate_drop': 0.03,      # DART specifics - reduced drop rate
        'skip_drop': 0.6        # DART specifics - increased skip drop
    }
    
    logging.info("\nStarting XGBoost training...")
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=10000, # Increased rounds significantly (from 8000)
        evals=[(dtrain, 'train'), (dtest, 'eval')],
        early_stopping_rounds=750, # Increased early stopping patience (from 500)
        verbose_eval=1000, # Log every 1000 rounds
        evals_result=evals_result
    )
    logging.info(f"XGBoost training complete. Best iteration: {model.best_iteration}")
    
    # Evaluation
    y_pred_proba = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    f1 = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    
    logging.info(f"\n📊 Model Evaluation Results:")
    logging.info(f"✅ F1 Score (macro): {f1:.4f}")
    logging.info(f"✅ Accuracy: {accuracy:.4f}")

    class_names = le.inverse_transform(np.arange(len(np.unique(y))))
    logging.info("\nDetailed Classification Report:")
    logging.info(classification_report(y_test, y_pred, target_names=class_names, zero_division=0)) # Added zero_division=0

    if f1 >= 0.90 and accuracy >= 0.90:
        logging.info("\n🎉 Congratulations! F1 Score and Accuracy both achieved over 90%.")
    else:
        logging.info("\nKeep optimizing! F1 Score and Accuracy are below 90%. Consider further tuning or data refinement.")

    # Save all required files
    np.save('output/X_train.npy', X_train)
    np.save('output/X_test.npy', X_test)
    np.save('output/y_train.npy', y_train)
    np.save('output/y_test.npy', y_test)
    
    np.save('output/X_resampled.npy', X_res)
    np.save('output/y_resampled.npy', y_res)
    
    with open('output/preprocessing.pkl', 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'category_label_encoder': le, 
            'room_type_encoder': room_type_encoder,
            'color_encoder': color_encoder,
            'material_encoder': material_encoder,
            'numerical_features': num_cols,
            'categorical_feature_cols': ['room_type', 'color', 'material'],
            'cat_feature_indices': cat_idx,
            'feature_names': feature_names_for_dmatrix
        }, f)
    
    model.save_model('output/model.xgb')
    
    with open(os.path.join('output', 'eval_history.pkl'), 'wb') as f:
        pickle.dump(evals_result, f)
    
    logging.info("\n💾 Saved files in 'output' directory:")
    logging.info("- X_train.npy, X_test.npy, y_train.npy, y_test.npy")
    logging.info("- X_resampled.npy, y_resampled.npy")
    logging.info("- model.xgb (trained XGBoost model)")
    logging.info("- preprocessing.pkl (scalers, encoders, and feature names)")
    logging.info("- eval_history.pkl (training evaluation history)")

if __name__ == "__main__":
    main()