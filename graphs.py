import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.preprocessing import LabelEncoder # Needed to inverse transform labels for report

# --- Configuration ---
OUTPUT_DIR = 'output'
EVAL_HISTORY_PATH = os.path.join(OUTPUT_DIR, 'eval_history.pkl')
Y_TEST_PATH = os.path.join(OUTPUT_DIR, 'y_test.npy')
Y_PRED_PATH = os.path.join(OUTPUT_DIR, 'y_pred_tuned.npy') # Ensure this matches your file
PREPROCESSING_PKL_PATH = os.path.join(OUTPUT_DIR, 'preprocessing.pkl') # To load LabelEncoder

# --- File Existence Checks ---
required_files = [EVAL_HISTORY_PATH, Y_TEST_PATH, Y_PRED_PATH, PREPROCESSING_PKL_PATH]
for f_path in required_files:
    if not os.path.exists(f_path):
        raise FileNotFoundError(f"Missing required file: {f_path}. Please ensure '1_data_preparation.py' ran successfully and generated all necessary outputs.")

# --- Load Data ---
print(f"Loading evaluation history from: {EVAL_HISTORY_PATH}")
with open(EVAL_HISTORY_PATH, 'rb') as f:
    evals_result = pickle.load(f)

print(f"Loading true labels from: {Y_TEST_PATH}")
y_test = np.load(Y_TEST_PATH)

print(f"Loading predicted labels from: {Y_PRED_PATH}")
y_pred = np.load(Y_PRED_PATH)

print(f"Loading preprocessing data from: {PREPROCESSING_PKL_PATH}")
with open(PREPROCESSING_PKL_PATH, 'rb') as f:
    preprocessing_data = pickle.load(f)
category_label_encoder = preprocessing_data['category_label_encoder']

# --- Extract Metrics for Plotting ---
epochs = range(len(evals_result['train']['mlogloss']))
train_mlogloss = evals_result['train']['mlogloss']
valid_mlogloss = evals_result['eval']['mlogloss']
train_merror = evals_result['train']['merror']
valid_merror = evals_result['eval']['merror']

# Convert error to accuracy
train_accuracy = [1 - e for e in train_merror]
valid_accuracy = [1 - e for e in valid_merror]

# Calculate final macro precision and recall on the test set
# These are single values, so they will be plotted as constant lines.
# To see progression, you'd need to modify 1_data_preparation.py to save these per iteration.
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average='macro', zero_division=0
)
precision_list = [macro_precision] * len(epochs)
recall_list = [macro_recall] * len(epochs)

# --- Plotting Function ---
def plot_metric(train_vals, valid_vals, title, ylabel, filename, include_valid=True):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_vals, label='Train', marker='o', markersize=4, linewidth=1.5)
    if include_valid:
        plt.plot(epochs, valid_vals, label='Validation', marker='x', markersize=4, linewidth=1.5)
    plt.xlabel('Iterations (Epochs)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Generated plot: {plot_path}")

# --- Generate Plots ---
print("\nGenerating plots...")
# 1. Training vs Validation Accuracy graph
plot_metric(train_accuracy, valid_accuracy, 'Training vs Validation Accuracy', 'Accuracy', 'accuracy_plot.png')

# 2. Training vs Validation Log Loss
plot_metric(train_mlogloss, valid_mlogloss, 'Training vs Validation Log Loss', 'Log Loss', 'logloss_plot.png')

# 3. Precision vs Validation Precision (as final macro average)
# Note: These will be flat lines as per-iteration precision is not captured by default.
# The 'Validation' line here will simply be the same as 'Train' since it's a single macro value.
plot_metric(precision_list, precision_list, 'Precision (Macro Average)', 'Precision', 'precision_plot.png', include_valid=False)

# 4. Recall vs Validation Recall (as final macro average)
# Similar to precision, this will be a flat line.
plot_metric(recall_list, recall_list, 'Recall (Macro Average)', 'Recall', 'recall_plot.png', include_valid=False)

print("\nAll plots generated successfully.")

# --- Generate Analysis Table/Report Performance ---
print("\nGenerating performance report...")
report_filename = os.path.join(OUTPUT_DIR, 'performance_report.txt')

with open(report_filename, 'w') as f:
    f.write("--- Model Performance Report ---\n")
    f.write(f"Date Generated: {pd.Timestamp.now()}\n\n")

    f.write("Overall Metrics (on Test Set):\n")
    f.write(f"  Macro F1 Score: {macro_f1:.4f}\n")
    f.write(f"  Macro Precision: {macro_precision:.4f}\n")
    f.write(f"  Macro Recall: {macro_recall:.4f}\n")
    
    # Accuracy is already printed by 1_data_preparation.py, but we can re-add it here for completeness
    # Or, if you want the accuracy from the last iteration of the test set from evals_result:
    last_valid_accuracy = valid_accuracy[-1] if valid_accuracy else 'N/A'
    f.write(f"  Test Set Accuracy (from evals_result): {last_valid_accuracy:.4f}\n\n")

    f.write("Detailed Classification Report (Per-Class Metrics on Test Set):\n\n")
    
    # Get class names for a readable report
    class_names = category_label_encoder.inverse_transform(np.arange(len(category_label_encoder.classes_)))
    
    report_text = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    f.write(report_text)
    f.write("\n\n--- End of Report ---\n")

print(f"✅ Performance report saved to: {report_filename}")
print("Please check the 'output' directory for the generated plots and the performance report.")