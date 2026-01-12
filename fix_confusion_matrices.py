"""
Fix Confusion Matrix Visualization
Regenerate confusion matrices with proper colors and visible numbers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import os

print("="*80)
print("FIXING CONFUSION MATRICES - Making Numbers Visible!")
print("="*80)

# Load label encoder
le = joblib.load('label_encoder_location.joblib')
class_names = le.classes_

# Function to create better confusion matrix
def create_fixed_confusion_matrix(y_true, y_pred, model_name, suffix=""):
    """Create confusion matrix with visible numbers"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    # Use a better colormap - Blues with white text on dark, black on light
    sns.heatmap(
        cm, 
        annot=True,           # Show numbers
        fmt='d',              # Integer format
        cmap='Blues',         # Blue colormap (good contrast)
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        annot_kws={'size': 14, 'weight': 'bold'},  # Larger, bold numbers
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    filename = f'CM_{model_name}_FIXED{suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()
    
    return cm

# Check which models have completed
models_to_fix = []

# Check for CSV files with predictions
if os.path.exists('Hasil_CrossValidation_CNNDNN_Default_GAN.csv'):
    models_to_fix.append(('CNN-DNN', 'CNNDNN_Default_GAN'))
    
if os.path.exists('Hasil_CrossValidation_CNNRNN_Default_GAN.csv'):
    models_to_fix.append(('CNN-RNN', 'CNNRNN_Default_GAN'))
    
if os.path.exists('Hasil_CrossValidation_CNN_BiLSTM_Attention_GAN.csv'):
    models_to_fix.append(('CNN-BiLSTM', 'CNN_BiLSTM_Attention_GAN'))

print(f"\nFound {len(models_to_fix)} completed models to fix")

# Load preprocessed data for reference
X = np.load('X_preprocessed.npy')
y = np.load('y_preprocessed.npy')

print(f"\nData loaded: {len(y)} samples")

# For each completed model, try to regenerate confusion matrix
for model_name, file_suffix in models_to_fix:
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Try to load the saved model
        model_file = f'best_model_{file_suffix.lower()}.h5'
        
        if os.path.exists(model_file):
            print(f"Loading model: {model_file}")
            from tensorflow import keras
            model = keras.models.load_model(model_file, compile=False)
            
            # Use last 20% as test set (same as training script)
            test_size = int(0.2 * len(X))
            X_test = X[-test_size:]
            y_test = y[-test_size:]
            
            print(f"Test set size: {len(y_test)}")
            
            # Make predictions
            y_pred_proba = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Create fixed confusion matrix
            cm = create_fixed_confusion_matrix(y_test, y_pred, model_name, f"_{file_suffix}")
            
            # Print accuracy
            accuracy = np.sum(y_pred == y_test) / len(y_test)
            print(f"Accuracy: {accuracy:.4f}")
            
        else:
            print(f"⚠️ Model file not found: {model_file}")
            print(f"   Skipping {model_name}")
            
    except Exception as e:
        print(f"❌ Error processing {model_name}: {e}")
        continue

print("\n" + "="*80)
print("✅ CONFUSION MATRIX FIX COMPLETE!")
print("="*80)
print("\nFixed confusion matrices saved with '_FIXED' suffix")
print("Look for files: CM_*_FIXED*.png")
