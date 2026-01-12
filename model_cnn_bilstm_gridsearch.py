"""
ECG Location Prediction - CNN-BiLSTM + Attention with GridSearch
Hyperparameter Tuning with Simplified Grid
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

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
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
print("CNN-BiLSTM + Attention - GridSearch Hyperparameter Tuning")
print("=" * 80)

# ============================================================================
# ATTENTION LAYER
# ============================================================================

class AttentionLayer(layers.Layer):
    """Attention mechanism for sequence data"""
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        e = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = inputs * a
        return tf.reduce_sum(output, axis=1)

# ============================================================================
# MODEL BUILDER
# ============================================================================

def build_cnn_bilstm_attention(input_shape, num_classes, 
                                learning_rate=0.001, 
                                dropout_rate=0.3,
                                lstm_units=128):
    """Build CNN-BiLSTM-Attention model with configurable hyperparameters"""
    
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
    
    # BiLSTM layers with configurable units
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Bidirectional(layers.LSTM(lstm_units//2, return_sequences=True))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Attention
    x = AttentionLayer()(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN_BiLSTM_Attention_GS')
    
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
    """
    Stage 1: Quick screening of all parameter combinations
    Uses single train-test split for speed
    """
    from sklearn.model_selection import train_test_split
    
    print("\n" + "=" * 80)
    print("STAGE 1: Quick Screening")
    print("=" * 80)
    
    # Generate all combinations
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    total_combinations = len(param_combinations)
    print(f"Testing {total_combinations} parameter combinations...")
    print(f"Method: Single train-test split (80/20)")
    
    # Single split for all combinations
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    all_results = []
    best_score = 0
    best_params = None
    
    for idx, param_values in enumerate(param_combinations):
        params = dict(zip(param_names, param_values))
        
        print(f"\n[{idx+1}/{total_combinations}] Testing: {params}")
        
        # Build and train model
        model = build_cnn_bilstm_attention(
            input_shape=X_train.shape[1:],
            num_classes=len(np.unique(y)),
            learning_rate=params.get('learning_rate', 0.001),
            dropout_rate=params.get('dropout_rate', 0.3),
            lstm_units=params.get('lstm_units', 128)
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,  # Quick screening - stop early
            restore_best_weights=True,
            verbose=0
        )
        
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,  # Reduced for quick screening
            batch_size=params.get('batch_size', 32),
            callbacks=[early_stop],
            verbose=0
        )
        
        train_time = time.time() - start_time
        
        # Evaluate
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
    
    # Sort results
    all_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\n" + "=" * 80)
    print("Quick Screening Complete!")
    print("=" * 80)
    print(f"\n🏆 Best Configuration:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"  Accuracy: {best_score:.4f}")
    
    print(f"\n📊 Top 5 Configurations:")
    for i, result in enumerate(all_results[:5]):
        print(f"\n{i+1}. Accuracy: {result['accuracy']:.4f}")
        print(f"   Params: {result['params']}")
    
    return best_params, all_results


def full_validation(X, y, best_params, n_splits=5):
    """
    Stage 2: Full cross-validation on best configuration
    """
    print("\n" + "=" * 80)
    print("STAGE 2: Full Cross-Validation on Best Configuration")
    print("=" * 80)
    print(f"\nBest Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\nPerforming {n_splits}-Fold Cross Validation...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{n_splits}")
        print(f"{'='*60}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Build model with best parameters
        model = build_cnn_bilstm_attention(
            input_shape=X_train.shape[1:],
            num_classes=len(np.unique(y)),
            learning_rate=best_params.get('learning_rate', 0.001),
            dropout_rate=best_params.get('dropout_rate', 0.3),
            lstm_units=best_params.get('lstm_units', 128)
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,  # Full validation - more patience
            restore_best_weights=True,
            verbose=0
        )
        
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,  # Full epochs for thorough validation
            batch_size=best_params.get('batch_size', 32),
            callbacks=[early_stop],
            verbose=1
        )
        
        train_time = time.time() - start_time
        
        # Evaluate
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
    
    # Calculate statistics
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
    
    # Load data
    X = np.load('X_preprocessed.npy')
    y = np.load('y_preprocessed.npy')
    
    # Load label encoder
    import joblib
    le = joblib.load('label_encoder_location.joblib')
    
    print(f"✓ Data shape: {X.shape}")
    print(f"✓ Labels shape: {y.shape}")
    print(f"✓ Classes: {le.classes_}")
    
    # Define parameter grid
    # You can expand this to 32 combinations if time permits
    param_grid = {
        'learning_rate': [0.001, 0.0001],      # 2 values
        'dropout_rate': [0.3, 0.5],            # 2 values
        'batch_size': [32, 64],                # 2 values
        'lstm_units': [128, 64]                # 2 values
    }
    # Total: 16 combinations
    
    # For 32 combinations, uncomment below:
    # param_grid = {
    #     'learning_rate': [0.001, 0.0001, 0.00001],  # 3 values
    #     'dropout_rate': [0.2, 0.3, 0.4, 0.5],       # 4 values
    #     'batch_size': [32, 64],                      # 2 values
    #     'lstm_units': [128, 96, 64]                  # 3 values
    # }
    # Total: 3 × 4 × 2 × 3 = 72 combinations (if you have time!)
    
    print("\n" + "=" * 80)
    print("2-STAGE GRIDSEARCH: CNN-BiLSTM + Attention")
    print("=" * 80)
    
    total_start = time.time()
    
    # STAGE 1: Quick Screening
    best_params, screening_results = quick_screening(X, y, param_grid)
    
    # Save screening results
    df_screening = pd.DataFrame([
        {**r['params'], 'accuracy': r['accuracy'], 'f1_score': r['f1_score']}
        for r in screening_results
    ])
    df_screening.to_csv('GridSearch_Stage1_Screening_BiLSTM.csv', index=False)
    print(f"\n✓ Stage 1 results saved to 'GridSearch_Stage1_Screening_BiLSTM.csv'")
    
    # STAGE 2: Full Validation
    final_results = full_validation(X, y, best_params, n_splits=5)
    
    # Save final results
    df_folds = pd.DataFrame(final_results['fold_results'])
    df_folds.to_csv('GridSearch_Stage2_FoldResults_BiLSTM.csv', index=False)
    
    # Save summary
    summary = {
        'Best Parameters': best_params,
        'Mean Accuracy': final_results['mean_accuracy'],
        'Std Accuracy': final_results['std_accuracy'],
        'Min Accuracy': final_results['min_accuracy'],
        'Max Accuracy': final_results['max_accuracy'],
        'Mean F1-Score': final_results['mean_f1'],
        'Total Time (hours)': (time.time() - total_start) / 3600
    }
    
    with open('GridSearch_Final_Summary_BiLSTM.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GridSearch Final Summary - CNN-BiLSTM + Attention\n")
        f.write("=" * 80 + "\n\n")
        f.write("BEST CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("CROSS-VALIDATION RESULTS:\n")
        f.write("-" * 80 + "\n")
        for key, value in summary.items():
            if key != 'Best Parameters':
                f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("FOLD-BY-FOLD RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(df_folds.to_string(index=False))
    
    print(f"\n✓ Final summary saved to 'GridSearch_Final_Summary_BiLSTM.txt'")
    
    # Visualization
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Screening results
    plt.subplot(2, 2, 1)
    x_pos = np.arange(len(screening_results))
    accuracies = [r['accuracy'] for r in screening_results]
    plt.bar(x_pos, accuracies, alpha=0.7, color='steelblue')
    plt.axhline(y=best_params, color='r', linestyle='--', label='Best')
    plt.xlabel('Configuration Index', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Stage 1: All Configurations Screening', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Fold results
    plt.subplot(2, 2, 2)
    fold_accs = [r['accuracy'] for r in final_results['fold_results']]
    folds = [r['fold'] for r in final_results['fold_results']]
    plt.bar(folds, fold_accs, alpha=0.7, color='forestgreen')
    plt.axhline(y=final_results['mean_accuracy'], color='r', linestyle='--', 
                label=f"Mean: {final_results['mean_accuracy']:.4f}")
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Stage 2: Best Config 5-Fold CV', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Parameter importance (learning rate)
    plt.subplot(2, 2, 3)
    lr_groups = df_screening.groupby('learning_rate')['accuracy'].mean()
    plt.bar(range(len(lr_groups)), lr_groups.values, alpha=0.7, color='darkorange')
    plt.xlabel('Learning Rate', fontsize=12)
    plt.ylabel('Mean Accuracy', fontsize=12)
    plt.title('Learning Rate Impact', fontsize=14, fontweight='bold')
    plt.xticks(range(len(lr_groups)), [f'{lr:.5f}' for lr in lr_groups.index])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Dropout impact
    plt.subplot(2, 2, 4)
    dropout_groups = df_screening.groupby('dropout_rate')['accuracy'].mean()
    plt.bar(range(len(dropout_groups)), dropout_groups.values, alpha=0.7, color='crimson')
    plt.xlabel('Dropout Rate', fontsize=12)
    plt.ylabel('Mean Accuracy', fontsize=12)
    plt.title('Dropout Rate Impact', fontsize=14, fontweight='bold')
    plt.xticks(range(len(dropout_groups)), [f'{d:.1f}' for d in dropout_groups.index])
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('GridSearch_Complete_Visualization_BiLSTM.png', dpi=300)
    print(f"✓ Visualization saved to 'GridSearch_Complete_Visualization_BiLSTM.png'")
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 80)
    print("✅ 2-STAGE GRIDSEARCH COMPLETED!")
    print("=" * 80)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Best accuracy: {final_results['mean_accuracy']:.4f} ± {final_results['std_accuracy']:.4f}")
    print(f"\nAll results saved!")

if __name__ == "__main__":
    main()

