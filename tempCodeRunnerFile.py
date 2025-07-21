import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score, accuracy_score, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
import os
import logging
from collections import defaultdict, Counter # Import Counter for better class distribution logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the plots directory exists
os.makedirs('plots', exist_ok=True)

# --- Updated plot_training_history function ---
def plot_training_history(filename_prefix, evals_result):
    """
    Plots training history metrics from XGBoost model using a directly provided evals_result dictionary.
    Args:
        filename_prefix (str): Prefix for saving the plot files.
        evals_result (dict): The dictionary containing evaluation history as returned by xgb.train.
    """
    if not evals_result:
        logging.warning("No evaluation history provided. Skipping training history plots.")
        return
    
    plt.figure(figsize=(14, 10))
    
    # 1. Accuracy Plot (using 'merror' if available, otherwise 'error')
    plt.subplot(2, 2, 1)
    has_accuracy_data = False
    for eval_set in evals_result:
        if 'merror' in evals_result[eval_set]:
            accuracy = [1 - x for x in evals_result[eval_set]['merror']]
            plt.plot(accuracy, label=f'{eval_set} accuracy')
            has_accuracy_data = True
        elif 'error' in evals_result[eval_set]:
            accuracy = [1 - x for x in evals_result[eval_set]['error']]
            plt.plot(accuracy, label=f'{eval_set} accuracy')
            has_accuracy_data = True
    
    if has_accuracy_data:
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Boosting Rounds')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.title('Training and Validation Accuracy (N/A)')
        plt.text(0.5, 0.5, 'No accuracy metric logged', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    
    # 2. Loss Plot (using 'mlogloss' if available, otherwise 'logloss')
    plt.subplot(2, 2, 2)
    has_loss_data = False
    for eval_set in evals_result:
        if 'mlogloss' in evals_result[eval_set]:
            plt.plot(evals_result[eval_set]['mlogloss'], label=f'{eval_set} loss')
            has_loss_data = True
        elif 'logloss' in evals_result[eval_set]:
            plt.plot(evals_result[eval_set]['logloss'], label=f'{eval_set} loss')
            has_loss_data = True
    
    if has_loss_data:
        plt.title('Training and Validation Loss')
        plt.xlabel('Boosting Rounds')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.title('Training and Validation Loss (N/A)')
        plt.text(0.5, 0.5, 'No loss metric logged', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    
    # Precision and Recall are not standard metrics directly stored in evals_result by default.
    plt.subplot(2, 2, 3)
    plt.title('Training and Validation Precision (N/A)')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    plt.text(0.5, 0.5, 'Requires custom callback logging', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    plt.subplot(2, 2, 4)
    plt.title('Training and Validation Recall (N/A)')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Recall')
    plt.grid(True, alpha=0.3)
    plt.text(0.5, 0.5, 'Requires custom callback logging', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Plot saved: {filename_prefix}_training_history.png")

def plot_feature_importance(booster, feature_names, filename):
    """
    Plots the top N feature importances from an XGBoost model.
    Args:
        booster (xgb.Booster): The trained XGBoost booster model.
        feature_names (list): A list of actual feature names corresponding to the
                               order of features in the input data.
        filename (str): The path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    importance = booster.get_score(importance_type='gain')
    
    mapped_importance = {}
    for k, v in importance.items():
        try:
            feature_idx = int(k[1:]) 
            if feature_idx < len(feature_names):
                mapped_importance[feature_names[feature_idx]] = v
            else:
                mapped_importance[k] = v 
        except (ValueError, IndexError):
            mapped_importance[k] = v

    importance_sorted = sorted(mapped_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    
    if not importance_sorted:
        logging.warning("No feature importances to plot. Skipping feature importance plot.")
        plt.close(fig)
        return

    features = [x[0] for x in importance_sorted]
    values = [x[1] for x in importance_sorted]
    
    sns.barplot(x=values, y=features, palette='viridis', ax=ax)
    plt.title('Top 20 Feature Importance (Gain)', fontsize=16)
    plt.xlabel('Gain', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Plot saved: {filename}")

def plot_confusion_matrix(y_true, y_pred, classes, filename, normalize=False):
    """
    Plots a confusion matrix.
    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels.
        classes (list): List of class names.
        filename (str): The path to save the plot.
        normalize (bool): Whether to normalize the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if len(classes) != cm.shape[0] or len(classes) != cm.shape[1]:
        logging.warning(f"Class names length ({len(classes)}) does not match confusion matrix dimensions ({cm.shape}). This might cause plotting issues.")
        # Attempt to map cm to class names, if necessary
        # This part requires a mapping of original labels to continuous 0..N-1 for proper indexing.
        # Assuming y_true and y_pred already contain labels 0..N-1 as per LabelEncoder.
        
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        cm = np.nan_to_num(cm, nan=0.0) 
    
    plt.figure(figsize=(min(len(classes)*0.8 + 2, 20), min(len(classes)*0.8 + 2, 18)))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=classes, yticklabels=classes, cmap='Blues',
                annot_kws={"size": 10}, cbar_kws={'shrink': 0.8}, linewidths=.5, linecolor='lightgray')
    plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix', 
              fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=min(10, 1000/len(classes)))
    plt.yticks(rotation=0, fontsize=min(10, 1000/len(classes)))
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Plot saved: {filename}")

def plot_precision_recall(y_true, y_scores, class_names, filename):
    """
    Plots Precision-Recall curves for each class.
    Args:
        y_true (np.array): True labels.
        y_scores (np.array): Predicted probabilities for each class.
        class_names (list): List of class names.
        filename (str): The path to save the plot.
    """
    num_classes = len(class_names)
    
    # Filter out classes that are not present in y_true
    unique_true_classes_idx = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    
    plt.figure(figsize=(14, 12))
    
    auc_scores = {}
    curves_to_plot = []

    for i in range(num_classes):
        class_name = class_names[i]
        
        if i not in unique_true_classes_idx:
            auc_scores[class_name] = np.nan
            continue

        if len(np.unique(y_scores[:, i])) < 2:
            logging.warning(f"Class '{class_name}' has constant probabilities or too few samples to compute PR curve. Setting AUC to NaN.")
            auc_scores[class_name] = np.nan
            continue

        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        
        if len(precision) > 1 and len(recall) > 1:
            auc_score = auc(recall, precision)
            auc_scores[class_name] = auc_score
            curves_to_plot.append((class_name, precision, recall, auc_score))
        else:
            logging.warning(f"Class '{class_name}' PR curve cannot be computed (too few points). Setting AUC to NaN.")
            auc_scores[class_name] = np.nan

    sorted_curves = sorted(curves_to_plot, key=lambda x: x[3] if not np.isnan(x[3]) else -1, reverse=True)
    
    if not sorted_curves:
        logging.warning("No valid Precision-Recall curves to plot. Skipping PR curve plot.")
        plt.close()
        return

    for class_name, precision, recall, auc_score in sorted_curves:
        plt.plot(recall, precision, lw=2, 
                 label=f'{class_name} (AUC = {auc_score:.2f})')

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve (Sorted by AUC)', fontsize=16)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=10) 
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Plot saved: {filename}")


def threshold_tuning_per_class(y_true, y_probs, class_names):
    """
    Performs threshold tuning for each class to maximize F1-score.
    Args:
        y_true (np.array): True labels.
        y_probs (np.array): Predicted probabilities for each class (from model.predict_proba).
        class_names (list): List of class names.
    Returns:
        np.array: An array of optimal thresholds for each class.
    """
    best_thresholds = np.zeros(len(class_names))
    
    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    
    logging.info("\n🔍 Threshold Tuning Results (Optimizing F1-score per class):")
    logging.info("{:<20} {:<15} {:<15} {:<15}".format(
        "Class", "Best Threshold", "F1 Score", "Support"))
    logging.info("-" * 65)
    
    for i in range(len(class_names)):
        best_f1 = -1 # Initialize with a value that any valid F1 will beat
        best_thresh = 0.5 # Default threshold
        
        true_labels_class = y_true_bin[:, i]
        probs_class = y_probs[:, i]
        
        support = np.sum(true_labels_class)
        
        if support == 0:
            logging.info(f"{class_names[i]:<20} {'N/A':<15} {'N/A':<15} {0:<15}")
            best_thresholds[i] = 0.5
            continue

        thresholds = np.unique(probs_class)
        # Ensure there are enough unique thresholds to sample from.
        # If not, fall back to a linspace.
        if len(thresholds) < 2:
            logging.warning(f"Class '{class_names[i]}' has constant probabilities ({np.unique(probs_class)[0]:.2f} or 0 unique values). Cannot tune threshold. Setting to 0.5.")
            best_thresholds[i] = 0.5
            continue
        
        # Limit the number of thresholds to check for performance, while also covering min/max.
        if len(thresholds) > 200: # Increased from 100 to 200 for finer granularity
            thresholds = np.linspace(np.min(probs_class), np.max(probs_class), 200)
        
        # Also include 0.0 and 1.0 explicitly, and sort
        thresholds = np.sort(np.unique(np.concatenate(([0.0], thresholds, [1.0]))))

        for thresh in thresholds:
            y_pred_thresh = (probs_class >= thresh).astype(int)
            f1 = f1_score(true_labels_class, y_pred_thresh, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                
        best_thresholds[i] = best_thresh
        logging.info("{:<20} {:<15.3f} {:<15.4f} {:<15}".format(
            class_names[i], best_thresh, best_f1, support))
    
    return best_thresholds

def main():
    output_dir = 'output'
    plots_dir = 'plots'
    
    os.makedirs(plots_dir, exist_ok=True)

    # --- Load all necessary files with robust error handling ---
    X_test_path = os.path.join(output_dir, 'X_test.npy')
    y_test_path = os.path.join(output_dir, 'y_test.npy')
    preprocessing_path = os.path.join(output_dir, 'preprocessing.pkl')
    model_path = os.path.join(output_dir, 'model.xgb')
    eval_history_path = os.path.join(output_dir, 'eval_history.pkl')

    try:
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)

        with open(preprocessing_path, 'rb') as f:
            prep = pickle.load(f)
        
        class_names = prep['category_label_encoder'].classes_
        feature_names = prep['feature_names'] 
        
        booster = xgb.Booster()
        booster.load_model(model_path)

        evals_result = {}
        if os.path.exists(eval_history_path):
            with open(eval_history_path, 'rb') as f:
                evals_result = pickle.load(f)
            logging.info("Successfully loaded model artifacts, test data, and training history.")
        else:
            logging.warning(f"Training evaluation history file not found at {eval_history_path}. Skipping history plots.")
            
    except FileNotFoundError as e:
        logging.error(f"Error: Required file not found: {e}. Please ensure the training script (1_data_p.py) has been run successfully and generated these files in the 'output/' directory.")
        return
    except Exception as e:
        logging.error(f"An error occurred during loading: {e}", exc_info=True)
        return

    # --- Plotting Training History ---
    plot_training_history(os.path.join(plots_dir, 'xgb'), evals_result)
    
    # --- Model Prediction and Initial Evaluation ---
    dtest = xgb.DMatrix(X_test, enable_categorical=True, feature_names=feature_names) 
    
    if hasattr(booster, 'best_iteration') and booster.best_iteration is not None:
        y_probs = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
        logging.info(f"Predicting with best iteration: {booster.best_iteration}")
    else:
        y_probs = booster.predict(dtest)
        logging.warning("booster.best_iteration not found, predicting with all trees.")

    y_pred_initial = np.argmax(y_probs, axis=1)

    f1_macro_initial = f1_score(y_test, y_pred_initial, average='macro', zero_division=0)
    f1_weighted_initial = f1_score(y_test, y_pred_initial, average='weighted', zero_division=0)
    accuracy_initial = accuracy_score(y_test, y_pred_initial)
    
    logging.info("\n📊 Initial Model Performance (Default Threshold):")
    logging.info(f"Accuracy: {accuracy_initial:.4f}")
    logging.info(f"F1 Macro: {f1_macro_initial:.4f}")
    logging.info(f"F1 Weighted: {f1_weighted_initial:.4f}")
    logging.info(f"Test Set Class Distribution: {Counter(y_test)}")
    logging.info(f"Predicted Class Distribution (Initial): {Counter(y_pred_initial)}")


    report = classification_report(
        y_test, y_pred_initial, 
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0
    )
    
    report_df = pd.DataFrame(report).transpose()
    logging.info("\n📈 Classification Report Summary (Default Threshold):")
    
    if 'accuracy' in report_df.index:
        report_df_display = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    else:
        report_df_display = report_df.drop(['macro avg', 'weighted avg'], errors='ignore')

    logging.info(report_df_display[['precision', 'recall', 'f1-score', 'support']]
                 .sort_values('f1-score', ascending=False)
                 .to_markdown(floatfmt=".3f"))
    
    report_output_path = os.path.join(plots_dir, 'classification_report_initial.md')
    with open(report_output_path, 'w') as f:
        f.write("# Initial Classification Report (Default Threshold)\n\n")
        f.write(report_df.to_markdown(floatfmt=".3f"))
    logging.info(f"Classification report saved: {report_output_path}")

    # --- Plotting various evaluation metrics ---
    plot_feature_importance(booster, feature_names, os.path.join(plots_dir, 'feature_importance.png'))
    plot_confusion_matrix(y_test, y_pred_initial, class_names, os.path.join(plots_dir, 'confusion_matrix.png'))
    plot_confusion_matrix(y_test, y_pred_initial, class_names, os.path.join(plots_dir, 'confusion_matrix_normalized.png'), normalize=True)
    plot_precision_recall(y_test, y_probs, class_names, os.path.join(plots_dir, 'precision_recall_curve.png'))
    
    # --- Threshold Tuning and Re-evaluation ---
    best_thresholds = threshold_tuning_per_class(y_test, y_probs, class_names)
    
    y_pred_tuned = np.zeros_like(y_test)
    for i in range(len(y_test)):
        adjusted_probs = y_probs[i] / (best_thresholds + 1e-9) 
        y_pred_tuned[i] = np.argmax(adjusted_probs)
    
    f1_macro_tuned = f1_score(y_test, y_pred_tuned, average='macro', zero_division=0)
    f1_weighted_tuned = f1_score(y_test, y_pred_tuned, average='weighted', zero_division=0)
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

    logging.info("\n🎯 Final Performance After Threshold Tuning:")
    logging.info(f"Accuracy: {accuracy_tuned:.4f} (Improvement: {accuracy_tuned - accuracy_initial:+.4f})")
    logging.info(f"F1 Macro: {f1_macro_tuned:.4f} (Improvement: {f1_macro_tuned - f1_macro_initial:+.4f})")
    logging.info(f"F1 Weighted: {f1_weighted_tuned:.4f} (Improvement: {f1_weighted_tuned - f1_weighted_initial:+.4f})")
    logging.info(f"Predicted Class Distribution (Tuned): {Counter(y_pred_tuned)}")

    report_tuned = classification_report(
        y_test, y_pred_tuned,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0
    )
    report_tuned_df = pd.DataFrame(report_tuned).transpose()
    logging.info("\n📈 Classification Report Summary (After Threshold Tuning):")

    if 'accuracy' in report_tuned_df.index:
        report_tuned_df_display = report_tuned_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    else:
        report_tuned_df_display = report_tuned_df.drop(['macro avg', 'weighted avg'], errors='ignore')

    logging.info(report_tuned_df_display[['precision', 'recall', 'f1-score', 'support']]
                 .sort_values('f1-score', ascending=False)
                 .to_markdown(floatfmt=".3f"))

    report_output_path_tuned = os.path.join(plots_dir, 'classification_report_tuned.md')
    with open(report_output_path_tuned, 'w') as f:
        f.write("# Classification Report (After Threshold Tuning)\n\n")
        f.write(report_tuned_df.to_markdown(floatfmt=".3f"))
    logging.info(f"Tuned classification report saved: {report_output_path_tuned}")


    # --- Save additional evaluation results ---
    np.save(os.path.join(output_dir, 'y_pred_tuned.npy'), y_pred_tuned)
    with open(os.path.join(output_dir, 'best_thresholds.pkl'), 'wb') as f:
        pickle.dump(best_thresholds, f)

    misclassified_indices = np.where(y_test != y_pred_tuned)[0]
    misclassified_filename = os.path.join(output_dir, 'misclassified_indices_tuned.npy')
    np.save(misclassified_filename, misclassified_indices)
    
    logging.info("\n💾 Saved additional files in 'output' directory:")
    logging.info(f"- {os.path.join(output_dir, 'y_pred_tuned.npy')} (predictions after threshold tuning)")
    logging.info(f"- {os.path.join(output_dir, 'best_thresholds.pkl')} (optimal thresholds per class)")
    logging.info(f"- {misclassified_filename} (indices of misclassified samples after tuning)")
    logging.info(f"All plots saved in '{plots_dir}/' directory.")

if __name__ == "__main__":
    main()