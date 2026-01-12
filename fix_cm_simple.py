"""
Generate Fixed Confusion Matrices for DNN and RNN
Simple script without emoji to avoid encoding issues
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

print("="*70)
print("GENERATING FIXED CONFUSION MATRICES")
print("="*70)

# Load data
print("\nLoading data...")
X = np.load('X_preprocessed.npy')
y = np.load('y_preprocessed.npy')
le = joblib.load('label_encoder_location.joblib')

print(f"Total samples: {len(y)}")
print(f"Classes: {le.classes_}")

# Use last 20% as test set
test_size = int(0.2 * len(X))
X_test = X[-test_size:]
y_test = y[-test_size:]

print(f"Test set size: {len(y_test)}")

# Function to create confusion matrix
def create_cm(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    # Use Blues colormap with proper annotation
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        cbar_kws={'label': 'Count'},
        annot_kws={'size': 14, 'weight': 'bold'},
        linewidths=0.5,
        linecolor='gray',
        vmin=0  # Ensure scale starts at 0
    )
    
    plt.title(f'Confusion Matrix - {model_name}\\nAccuracy: {acc:.4f}',
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    filename = f'CM_{model_name.replace(" ", "_").replace("-", "")}_FIXED.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename} (Accuracy: {acc:.4f})")
    plt.close()

# Process DNN
print("\n" + "="*70)
print("Processing CNN-DNN...")
print("="*70)

try:
    from tensorflow import keras
    
    # Try different possible paths
    dnn_paths = [
        'best_model_cnn_dnn_gan.h5',
        'dnn default/best_model_cnndnn_default_gan.h5',
        'best_model_cnndnn_default_gan.h5'
    ]
    
    model_dnn = None
    for path in dnn_paths:
        try:
            model_dnn = keras.models.load_model(path, compile=False)
            print(f"Loaded model from: {path}")
            break
        except:
            continue
    
    if model_dnn:
        y_pred_dnn = np.argmax(model_dnn.predict(X_test, verbose=0), axis=1)
        create_cm(y_test, y_pred_dnn, "CNN-DNN")
    else:
        print("ERROR: Could not find DNN model file")
        
except Exception as e:
    print(f"ERROR processing DNN: {e}")

# Process RNN
print("\n" + "="*70)
print("Processing CNN-RNN...")
print("="*70)

try:
    # Try different possible paths
    rnn_paths = [
        'best_model_cnnrnn_default_gan.h5',
        'rnn default/best_model_cnnrnn_default_gan.h5',
        'best_model_cnn_rnn_gan.h5'
    ]
    
    model_rnn = None
    for path in rnn_paths:
        try:
            model_rnn = keras.models.load_model(path, compile=False)
            print(f"Loaded model from: {path}")
            break
        except:
            continue
    
    if model_rnn:
        y_pred_rnn = np.argmax(model_rnn.predict(X_test, verbose=0), axis=1)
        create_cm(y_test, y_pred_rnn, "CNN-RNN")
    else:
        print("ERROR: Could not find RNN model file")
        
except Exception as e:
    print(f"ERROR processing RNN: {e}")

print("\n" + "="*70)
print("DONE!")
print("="*70)
print("\nLook for files: CM_*_FIXED.png")
