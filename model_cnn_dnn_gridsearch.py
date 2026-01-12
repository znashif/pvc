"""
ECG Location Prediction - CNN-DNN with 2-Stage GridSearch
Stage 1: Quick screening with low epochs
Stage 2: Full validation on best config
"""

import numpy as np
import pandas as pd
import os
import time
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import itertools

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# GPU Config
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

print("=" * 80)
print("CNN-DNN - 2-Stage GridSearch")
print("=" * 80)

# ============================================================================
# MODEL BUILDER
# ============================================================================

def build_cnn_dnn(input_shape, num_classes, 
                  learning_rate=0.001, 
                  dropout_rate=0.3,
                  dense_units=512):
    """Build CNN-DNN model"""
    
    inputs = layers.Input(shape=input_shape)
    
    # CNN layers
    x = layers.Conv1D(64, kernel_size=7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv1D(128, kernel_size=5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv1D(256, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Flatten
    x = layers.Flatten()(x)
    
    # DNN layers
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(dense_units//2, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN_DNN_GS')
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================================================
# 2-STAGE GRIDSEARCH
# ============================================================================

def quick_screening(X, y, param_grid):
    """Stage 1: Quick screening with 20 epochs"""
    
    print("\n" + "=" * 80)
    print("STAGE 1: Quick Screening (20 epochs)")
    print("=" * 80)
    
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    total_combinations = len(param_combinations)
    print(f"Testing {total_combinations} parameter combinations...")
    
    # Single split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    all_results = []
    best_score = 0
    best_params = None
    
    for idx, param_values in enumerate(param_combinations):
        params = dict(zip(param_names, param_values))
        
        print(f"\n[{idx+1}/{total_combinations}] Testing: {params}")
        
        model = build_cnn_dnn(
            input_shape=X_train.shape[1:],
            num_classes=len(np.unique(y)),
            learning_rate=params.get('learning_rate', 0.001),
            dropout_rate=params.get('dropout_rate', 0.3),
            dense_units=params.get('dense_units', 512)
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
        
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,  # Quick screening
            batch_size=params.get('batch_size', 32),
            callbacks=[early_stop],
            verbose=0
        )
        
        train_time = time.time() - start_time
        
        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        
        result = {
            'params': params,
            'accuracy': accuracy,
            'f1_score': f1,
            'train_time': train_time
        }
        all_results.append(result)
        
        print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Time: {train_time:.1f}s")
        
        if accuracy > best_score:
            best_score = accuracy
            best_params = params
            print(f"  ⭐ New best!")
    
    all_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\n" + "=" * 80)
    print("Quick Screening Complete!")
    print("=" * 80)
    print(f"\n🏆 Best Configuration:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"  Accuracy: {best_score:.4f}")
    
    return best_params, all_results


def full_validation(X, y, best_params, n_splits=5):
    """Stage 2: Full 5-Fold CV with 50 epochs"""
    
    print("\n" + "=" * 80)
    print("STAGE 2: Full 5-Fold Cross-Validation (50 epochs)")
    print("=" * 80)
    print(f"\nBest Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{n_splits}")
        print(f"{'='*60}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = build_cnn_dnn(
            input_shape=X_train.shape[1:],
            num_classes=len(np.unique(y)),
            learning_rate=best_params.get('learning_rate', 0.001),
            dropout_rate=best_params.get('dropout_rate', 0.3),
            dense_units=best_params.get('dense_units', 512)
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,  # Full validation
            batch_size=best_params.get('batch_size', 32),
            callbacks=[early_stop],
            verbose=1
        )
        
        train_time = time.time() - start_time
        
        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'f1_score': f1,
            'train_time': train_time
        })
        
        print(f"\nFold {fold+1} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Time: {train_time:.1f}s")
    
    accuracies = [r['accuracy'] for r in fold_results]
    f1_scores = [r['f1_score'] for r in fold_results]
    
    final_results = {
        'best_params': best_params,
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'min_accuracy': np.min(accuracies),
        'max_accuracy': np.max(accuracies),
        'mean_f1': np.mean(f1_scores),
        'std_f1': np.std(f1_scores),
        'fold_results': fold_results
    }
    
    print("\n" + "=" * 80)
    print("Full Validation Complete!")
    print("=" * 80)
    print(f"\n📊 Final Results:")
    print(f"  Mean Accuracy: {final_results['mean_accuracy']:.4f} ± {final_results['std_accuracy']:.4f}")
    print(f"  Min Accuracy: {final_results['min_accuracy']:.4f}")
    print(f"  Max Accuracy: {final_results['max_accuracy']:.4f}")
    print(f"  Mean F1-Score: {final_results['mean_f1']:.4f} ± {final_results['std_f1']:.4f}")
    
    return final_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n📂 Loading preprocessed data...")
    
    X = np.load('X_preprocessed.npy')
    y = np.load('y_preprocessed.npy')
    
    import joblib
    le = joblib.load('label_encoder_location.joblib')
    
    print(f"✓ Data shape: {X.shape}")
    print(f"✓ Labels shape: {y.shape}")
    print(f"✓ Classes: {le.classes_}")
    
    param_grid = {
        'learning_rate': [0.001, 0.0001],
        'dropout_rate': [0.3, 0.5],
        'batch_size': [32, 64],
        'dense_units': [512, 256]
    }
    
    print("\n" + "=" * 80)
    print("2-STAGE GRIDSEARCH: CNN-DNN")
    print("=" * 80)
    
    total_start = time.time()
    
    # STAGE 1
    best_params, screening_results = quick_screening(X, y, param_grid)
    
    df_screening = pd.DataFrame([
        {**r['params'], 'accuracy': r['accuracy'], 'f1_score': r['f1_score']}
        for r in screening_results
    ])
    df_screening.to_csv('GridSearch_Stage1_Screening_DNN.csv', index=False)
    print(f"\n✓ Stage 1 results saved")
    
    # STAGE 2
    final_results = full_validation(X, y, best_params, n_splits=5)
    
    df_folds = pd.DataFrame(final_results['fold_results'])
    df_folds.to_csv('GridSearch_Stage2_FoldResults_DNN.csv', index=False)
    
    # Save summary
    with open('GridSearch_Final_Summary_DNN.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GridSearch Final Summary - CNN-DNN\n")
        f.write("=" * 80 + "\n\n")
        f.write("BEST CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nMean Accuracy: {final_results['mean_accuracy']:.4f} ± {final_results['std_accuracy']:.4f}\n")
        f.write(f"Mean F1-Score: {final_results['mean_f1']:.4f}\n")
    
    print(f"\n✓ Summary saved")
    
    # Visualization
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    x_pos = np.arange(len(screening_results))
    accuracies = [r['accuracy'] for r in screening_results]
    plt.bar(x_pos, accuracies, alpha=0.7, color='forestgreen')
    plt.xlabel('Configuration Index')
    plt.ylabel('Accuracy')
    plt.title('Stage 1: All Configurations - DNN', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(2, 2, 2)
    fold_accs = [r['accuracy'] for r in final_results['fold_results']]
    folds = [r['fold'] for r in final_results['fold_results']]
    plt.bar(folds, fold_accs, alpha=0.7, color='darkorange')
    plt.axhline(y=final_results['mean_accuracy'], color='r', linestyle='--')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Stage 2: Best Config 5-Fold CV - DNN', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(2, 2, 3)
    lr_groups = df_screening.groupby('learning_rate')['accuracy'].mean()
    plt.bar(range(len(lr_groups)), lr_groups.values, alpha=0.7, color='steelblue')
    plt.xlabel('Learning Rate')
    plt.ylabel('Mean Accuracy')
    plt.title('Learning Rate Impact - DNN', fontweight='bold')
    plt.xticks(range(len(lr_groups)), [f'{lr:.5f}' for lr in lr_groups.index])
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(2, 2, 4)
    dropout_groups = df_screening.groupby('dropout_rate')['accuracy'].mean()
    plt.bar(range(len(dropout_groups)), dropout_groups.values, alpha=0.7, color='crimson')
    plt.xlabel('Dropout Rate')
    plt.ylabel('Mean Accuracy')
    plt.title('Dropout Impact - DNN', fontweight='bold')
    plt.xticks(range(len(dropout_groups)), [f'{d:.1f}' for d in dropout_groups.index])
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('GridSearch_Complete_Visualization_DNN.png', dpi=300)
    print(f"✓ Visualization saved")
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 80)
    print("✅ 2-STAGE GRIDSEARCH COMPLETED!")
    print("=" * 80)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Best accuracy: {final_results['mean_accuracy']:.4f} ± {final_results['std_accuracy']:.4f}")

if __name__ == "__main__":
    main()
