"""
ECG Location Prediction - CNN-BiLSTM + Attention Model
4 Classes: Left_Ventricular, Right_Ventricular, Outflow_Tract, Septal
"""

import numpy as np
import pandas as pd
import os
import time
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt
import seaborn as sns

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# GPU Config
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

print("=" * 80)
print("CNN-BiLSTM + Attention Model for ECG Location Prediction")
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
# MODEL ARCHITECTURE
# ============================================================================

def build_cnn_bilstm_attention(input_shape, num_classes):
    """Build CNN-BiLSTM-Attention model"""
    inputs = layers.Input(shape=input_shape)
    
    # CNN layers
    x = layers.Conv1D(64, kernel_size=7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(128, kernel_size=5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(256, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    
    # BiLSTM layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    
    # Attention
    x = AttentionLayer()(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN_BiLSTM_Attention')
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model

def calculate_specificity(y_true, y_pred, num_classes):
    """Calculate specificity for each class"""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    specificity = []
    
    for i in range(num_classes):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity.append(spec)
    
    return np.array(specificity)

def plot_bilstm_report(history, model, X_test, y_test, fold_no, class_names):
    """Generate comprehensive report plots for CNN-BiLSTM"""
    suffix = "_CNN_BiLSTM_Attention_GAN"
    
    # 1. Learning Curve
    print("Generating Learning Curve...")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val', linewidth=2)
    plt.title(f'Accuracy Fold {fold_no}', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val', linewidth=2)
    plt.title(f'Loss Fold {fold_no}', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Learning_Curve{suffix}_Fold_{fold_no}.png', dpi=300)
    plt.close()
    
    # 2. Confusion Matrix
    print("Generating Confusion Matrix...")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - CNN-BiLSTM+Attention', fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'CM{suffix}.png', dpi=300)
    plt.close()
    
    # 3. ROC Curve
    print("Generating ROC Curve...")
    y_pred_probs = model.predict(X_test, verbose=0)
    num_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=range(num_classes))
    
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{class_names[i]} (AUC={roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - CNN-BiLSTM+Attention', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'ROC{suffix}.png', dpi=300)
    plt.close()
    
    # 4. Attention Map (if attention layer exists)
    try:
        print("Generating Attention Map...")
        # Try to find attention layer
        attention_layer = None
        for layer in model.layers:
            if 'attention' in layer.name.lower():
                attention_layer = layer
                break
        
        if attention_layer is not None:
            # Get intermediate model up to attention layer
            for i, layer in enumerate(model.layers):
                if layer == attention_layer:
                    # Get output from the layer before attention
                    att_input_layer = model.layers[i-1]
                    att_model = Model(inputs=model.input, outputs=att_input_layer.output)
                    att_features = att_model.predict(X_test[0:1], verbose=0)
                    
                    plt.figure(figsize=(12, 4))
                    if len(att_features.shape) == 3:
                        # Plot attention weights across timesteps
                        att_weights = np.mean(att_features[0], axis=1)
                        plt.plot(att_weights, color='blue', linewidth=2)
                        plt.title('Attention Weights - CNN-BiLSTM+Attention', fontweight='bold')
                        plt.xlabel('Timestep')
                        plt.ylabel('Attention Weight')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(f'Attention{suffix}.png', dpi=300)
                        plt.close()
                    break
        else:
            print("  Attention layer not found, skipping...")
    except Exception as e:
        print(f"  Skipping Attention Map: {e}")
    
    # 5. t-SNE
    try:
        print("Generating t-SNE...")
        # Get features from second-to-last layer
        feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
        features = feature_model.predict(X_test, verbose=0)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_test)-1))
        embeddings = tsne.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                            c=y_test, cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, ticks=range(num_classes), 
                    label='Class', boundaries=np.arange(num_classes+1)-0.5)
        plt.title('t-SNE Visualization - CNN-BiLSTM+Attention', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'TSNE{suffix}.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"  Skipping t-SNE: {e}")
    
    print("✓ All visualizations generated successfully")


# ============================================================================
# MAIN TRAINING PIPELINE
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
    
    # K-Fold Cross Validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    class_metrics_list = []
    best_acc = 0
    best_model = None
    X_test_best = None
    y_test_best = None
    history_best = None
    
    print(f"\n🔄 Starting {n_splits}-Fold Cross Validation...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*80}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"{'='*80}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Build model
        model = build_cnn_bilstm_attention(
            input_shape=X_train.shape[1:],
            num_classes=len(le.classes_)
        )
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            f'model_cnn_bilstm_fold{fold+1}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=0
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0
        )
        
        # Train
        print(f"🏋️ Training model...")
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.15,
            epochs=50,
            batch_size=32,
            callbacks=[checkpoint, early_stop, reduce_lr],
            verbose=1
        )
        
        train_time = time.time() - start_time
        
        # Load best model
        model.load_weights(f'model_cnn_bilstm_fold{fold+1}.h5')
        
        # Evaluate
        print(f"\n📊 Evaluating model...")
        start_inf = time.time()
        y_pred_proba = model.predict(X_test, verbose=0)
        inf_time = time.time() - start_inf
        
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )
        specificity = calculate_specificity(y_test, y_pred, len(le.classes_))
        
        # Store results
        fold_results.append({
            'Fold': fold + 1,
            'Akurasi': accuracy,
            'F1_Score': f1_macro,
            'F1_Weighted': f1_weighted,
            'Train_Time': train_time,
            'Inf_Time': inf_time
        })
        
        class_metrics_list.append({
            'p': precision,
            'r': recall,
            's': specificity,
            'f1': f1
        })
        
        print(f"\n✓ Fold {fold + 1} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score (Macro): {f1_macro:.4f}")
        print(f"  Train Time: {train_time:.2f}s")
        print(f"  Inference Time: {inf_time:.4f}s")
        
        # Save best model
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = model
            X_test_best = X_test
            y_test_best = y_test
            history_best = history
            model.save('best_model_cnn_bilstm_attention_gan.h5')
            print(f"  ⭐ New best model saved!")
    
    # ==========================================
    # GENERATE TABLES & PLOTS
    # ==========================================
    print("\n" + "="*80)
    print("📊 Generating Report for CNN-BiLSTM + Attention...")
    print("="*80)
    
    plot_bilstm_report(history_best, best_model, X_test_best, y_test_best, 0, le.classes_)
    
    # ==========================================
    # ADDITIONAL COMPREHENSIVE PLOTS
    # ==========================================
    
    # Plot 1: Fold Comparison - Accuracy
    plt.figure(figsize=(12, 6))
    folds = [r['Fold'] for r in fold_results]
    accuracies = [r['Akurasi'] for r in fold_results]
    plt.bar(folds, accuracies, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axhline(y=np.mean(accuracies), color='r', linestyle='--', label=f'Mean: {np.mean(accuracies):.4f}')
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy Comparison Across Folds - CNN-BiLSTM+Attention', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('Grafik_Perbandingan_Fold_CNN_BiLSTM_Attention.png', dpi=300)
    plt.close()
    print("✓ Grafik Perbandingan Fold")
    
    # Plot 2: Box Plot - Consistency
    plt.figure(figsize=(10, 6))
    plt.boxplot([accuracies], labels=['CNN-BiLSTM+Attention'], patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Consistency - CNN-BiLSTM+Attention', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('Grafik_Konsistensi_CNN_BiLSTM_Attention.png', dpi=300)
    plt.close()
    print("✓ Grafik Konsistensi")
    
    # Plot 3: Training Time vs Accuracy
    plt.figure(figsize=(10, 6))
    train_times = [r['Train_Time'] for r in fold_results]
    plt.scatter(train_times, accuracies, s=100, alpha=0.6, c=range(len(folds)), cmap='viridis')
    for i, fold in enumerate(folds):
        plt.annotate(f'Fold {fold}', (train_times[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    plt.xlabel('Training Time (s)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training Time vs Accuracy - CNN-BiLSTM+Attention', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Grafik_Time_vs_Accuracy_CNN_BiLSTM_Attention.png', dpi=300)
    plt.close()
    print("✓ Grafik Training Time vs Accuracy")
    
    # Table 1: Overall Evaluation
    df_res = pd.DataFrame(fold_results)
    t1 = pd.DataFrame({
        'Metric': ['Rata-rata Akurasi', 'Akurasi Terbaik', 'Rata-rata F1'],
        'Value': [df_res['Akurasi'].mean(), best_acc, df_res['F1_Score'].mean()]
    })
    t1.to_csv("Tabel_1_Evaluasi_CNN_BiLSTM_Attention_GAN.csv", index=False)
    print("✓ Tabel 1: Evaluasi Model")
    
    # Table 2: Per-Class Metrics
    avg_p = np.mean([m['p'] for m in class_metrics_list], axis=0)
    avg_r = np.mean([m['r'] for m in class_metrics_list], axis=0)
    avg_s = np.mean([m['s'] for m in class_metrics_list], axis=0)
    avg_f1 = np.mean([m['f1'] for m in class_metrics_list], axis=0)
    
    t2 = pd.DataFrame({
        'Kelas': le.classes_,
        'Recall': avg_r,
        'Spesifisitas': avg_s,
        'Presisi': avg_p,
        'F1': avg_f1
    })
    t2.to_csv("Tabel_2_Detail_Kelas_CNN_BiLSTM_Attention_GAN.csv", index=False)
    print("✓ Tabel 2: Detail Per Kelas")
    
    # Table 3: Efficiency
    params = best_model.count_params()
    size = os.path.getsize("best_model_cnn_bilstm_attention_gan.h5") / (1024*1024)
    t3 = pd.DataFrame({
        'Metric': ['Total Parameter', 'Size (MB)', 'Avg Train Time (s)', 'Avg Inf Time (s)'],
        'Value': [params, size, df_res['Train_Time'].mean(), (df_res['Inf_Time'].mean()/len(X_test_best))]
    })
    t3.to_csv("Tabel_3_Efisiensi_CNN_BiLSTM_Attention_GAN.csv", index=False)
    print("✓ Tabel 3: Efisiensi Model")
    
    # Table 4: Consistency
    t4 = pd.DataFrame({
        'Metric': ['Avg Accuracy', 'Std Dev', 'Min', 'Max'],
        'Value': [df_res['Akurasi'].mean(), df_res['Akurasi'].std(), 
                  df_res['Akurasi'].min(), df_res['Akurasi'].max()]
    })
    t4.to_csv("Tabel_4_Konsistensi_CNN_BiLSTM_Attention_GAN.csv", index=False)
    print("✓ Tabel 4: Konsistensi Model")
    
    # Save fold results
    df_res.to_csv("Hasil_CrossValidation_CNN_BiLSTM_Attention_GAN.csv", index=False)
    print("✓ Hasil Cross-Validation")
    
    print("\n" + "="*80)
    print("✅ CNN-BiLSTM + Attention (GAN) SELESAI")
    print("="*80)
    print(f"📁 Semua output tersimpan dengan suffix '_CNN_BiLSTM_Attention_GAN'")
    print(f"🏆 Best Accuracy: {best_acc:.4f}")
    print(f"📊 Average Accuracy: {df_res['Akurasi'].mean():.4f} ± {df_res['Akurasi'].std():.4f}")

if __name__ == "__main__":
    main()
