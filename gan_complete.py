# ==========================================
# CNN-BiLSTM-ATTENTION WITH ECG GAN
# COMPLETE VERSION WITH ALL TABLES & PLOTS
# ==========================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, recall_score, precision_score, 
    roc_curve, auc, roc_auc_score, matthews_corrcoef
)
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, BatchNormalization,
    Dense, Dropout, LSTM, Bidirectional,
    LeakyReLU, GlobalAveragePooling1D, Flatten,
    Add, LayerNormalization, GRU, GlobalMaxPooling1D,
    Concatenate, Reshape, Multiply, Lambda,
    UpSampling1D, Conv1DTranspose, Cropping1D, ZeroPadding1D
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import mixed_precision
import gc
import warnings
warnings.filterwarnings('ignore')
from tqdm.auto import tqdm

print("="*80)
print("🚀 CNN-BiLSTM-ATTENTION WITH ECG GAN - COMPLETE VERSION")
print("="*80)
print(f"TensorFlow Version: {tf.__version__}")
print(f"NumPy Version: {np.__version__}")

# ==========================================
# GPU CONFIGURATION
# ==========================================
print("\n" + "="*80)
print("🔧 GPU CONFIGURATION")
print("="*80)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        
        print(f"✅ GPU ENABLED: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        print(f"✅ Mixed Precision ENABLED (float16)")
        print(f"✅ Memory Growth ENABLED")
        
        BATCH_SIZE = 64
        
    except RuntimeError as e:
        print(f"⚠️  GPU configuration error: {e}")
        BATCH_SIZE = 32
else:
    print("⚠️  NO GPU DETECTED - Using CPU")
    BATCH_SIZE = 32

# ==========================================
# SEED
# ==========================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================================
# LOAD DATA
# ==========================================
print("\n" + "="*80)
print("📂 LOADING DATA")
print("="*80)

try:
    X_raw = np.load('X_raw.npy')
    y_raw = np.load('y_raw.npy')
    patient_ids = np.load('patient_ids.npy')
    encoder = joblib.load('label_encoder.joblib')
    print("✅ Loaded existing processed data")
except:
    print("⚠️  Creating sample data for testing...")
    n_samples = 5000  # Increased for better class distribution
    seq_len = 1000
    n_channels = 12
    X_raw = np.random.randn(n_samples, seq_len, n_channels).astype(np.float32)
    
    # Create balanced classes - ensure each class has enough samples
    n_classes = 4
    samples_per_class = n_samples // n_classes
    y_raw = np.concatenate([np.full(samples_per_class, i) for i in range(n_classes)])
    # Add remaining samples to last class
    remaining = n_samples - len(y_raw)
    if remaining > 0:
        y_raw = np.concatenate([y_raw, np.full(remaining, n_classes-1)])
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X_raw = X_raw[shuffle_idx]
    y_raw = y_raw[shuffle_idx]
    
    patient_ids = np.random.randint(0, 50, n_samples)
    
    encoder = LabelEncoder()
    encoder.classes_ = np.array(['Left_Ventricular_Region', 'Outflow_Tract_Region', 
                                'Right_Ventricular_Region', 'Septal_Region'])
    print("✅ Created sample data")

class_names = encoder.classes_
NUM_CLASSES = len(class_names)

print(f"\n📊 Data Summary:")
print(f"   Samples: {X_raw.shape[0]}")
print(f"   Sequence Length: {X_raw.shape[1]}")
print(f"   Channels: {X_raw.shape[2]}")
print(f"   Patients: {len(np.unique(patient_ids))}")
print(f"   Classes: {NUM_CLASSES} - {list(class_names)}")

# ==========================================
# CNN-BiLSTM-ATTENTION MODEL
# ==========================================
def build_cnn_bilstm_attention_model(input_shape, num_classes):
    """CNN-BiLSTM dengan Attention Mechanism"""
    
    inputs = Input(shape=input_shape, name='input_layer')
    
    # CNN Feature Extractor
    x = Conv1D(32, 7, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(64, 5, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    # Bidirectional LSTM
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    
    # Attention Mechanism
    attention_probs = Dense(1, activation='tanh', name='attention_dense')(x)
    attention_probs = tf.keras.layers.Softmax(axis=1, name='attention_weights')(attention_probs)
    attention_mul = Multiply()([x, attention_probs])
    
    # Global Pooling
    x = GlobalAveragePooling1D()(attention_mul)
    
    # Dense Layers (embedding layer for t-SNE)
    x = Dense(128, activation='relu', name='embedding')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output Layer - dtype='float32' handles mixed precision automatically
    outputs = Dense(num_classes, activation='softmax', dtype='float32', name='output')(x)
    
    model = Model(inputs, outputs, name='CNN_BiLSTM_Attention')
    
    return model

# ==========================================
# ECG GAN - SIMPLIFIED (NO LAMBDA)
# ==========================================
def build_ecg_gan(input_shape, latent_dim=100):
    """ECG GAN untuk data augmentation - Simplified version"""
    
    seq_len, n_channels = input_shape
    
    # Generator - Direct output to target shape
    generator_input = Input(shape=(latent_dim,))
    
    # Calculate to get exact seq_len after upsampling
    # We'll use 4 upsampling layers with stride 2 each
    # So we need initial_len such that initial_len * 2^4 = seq_len
    initial_len = seq_len // 16
    
    x = Dense(64 * initial_len)(generator_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((initial_len, 64))(x)
    
    # Upsample to 2x
    x = Conv1DTranspose(32, 5, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Upsample to 4x
    x = Conv1DTranspose(16, 5, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Upsample to 8x
    x = Conv1DTranspose(8, 5, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Upsample to 16x (final)
    x = Conv1DTranspose(n_channels, 5, strides=2, padding='same', activation='tanh')(x)
    
    generator = Model(generator_input, x, name='Generator')
    
    # Discriminator
    discriminator_input = Input(shape=input_shape)
    
    x = Conv1D(32, 5, padding='same')(discriminator_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(64, 5, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(128, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = GlobalAveragePooling1D()(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    discriminator_output = Dense(1, activation='sigmoid', dtype='float32', name='discriminator_output')(x)
    
    discriminator = Model(discriminator_input, discriminator_output, name='Discriminator')
    
    return generator, discriminator

# ==========================================
# TRAIN ECG GAN
# ==========================================
def train_ecg_gan(generator, discriminator, X_train, epochs=30, batch_size=None, latent_dim=100):
    """Train ECG GAN"""
    
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    print(f"\n🎨 Training ECG GAN...")
    print(f"   Epochs: {epochs}, Batch Size: {batch_size}")
    
    discriminator.compile(
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    generated_ecg = generator(gan_input)
    gan_output = discriminator(generated_ecg)
    
    gan = Model(gan_input, gan_output, name='GAN')
    gan.compile(
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        loss='binary_crossentropy'
    )
    
    X_min, X_max = X_train.min(), X_train.max()
    X_train_gan = (X_train - X_min) / (X_max - X_min) * 2 - 1
    
    n_batches = len(X_train_gan) // batch_size
    
    for epoch in range(epochs):
        d_losses = []
        g_losses = []
        
        for _ in range(min(n_batches, 10)):
            idx = np.random.randint(0, X_train_gan.shape[0], batch_size)
            real_ecgs = X_train_gan[idx]
            
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_ecgs = generator.predict(noise, verbose=0)
            
            real_labels = np.ones((batch_size, 1)) * 0.9
            fake_labels = np.zeros((batch_size, 1)) + 0.1
            
            d_loss_real = discriminator.train_on_batch(real_ecgs, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_ecgs, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
            d_losses.append(d_loss)
            
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
            g_losses.append(g_loss)
        
        if epoch % 5 == 0:
            print(f"   Epoch {epoch}/{epochs}: D_loss={np.mean(d_losses):.4f}, G_loss={np.mean(g_losses):.4f}")
        
        if epoch % 10 == 0:
            gc.collect()
    
    print("✅ ECG GAN training completed")
    
    del gan
    gc.collect()
    
    return generator

# ==========================================
# AUGMENT WITH GAN
# ==========================================
def augment_with_gan(generator, X_train, y_train, n_samples_per_class=500, latent_dim=100):
    """Augment data using trained GAN"""
    
    print(f"\n🔧 Augmenting data with GAN...")
    
    unique_classes = np.unique(y_train)
    X_augmented = [X_train]
    y_augmented = [y_train]
    
    for class_idx in unique_classes:
        class_mask = y_train == class_idx
        X_class = X_train[class_mask]
        n_current = len(X_class)
        
        print(f"   Class {class_idx}: {n_current} samples", end="")
        
        if n_current < n_samples_per_class:
            n_needed = n_samples_per_class - n_current
            print(f" -> generating {n_needed} samples")
            
            generated_samples = []
            n_batches = (n_needed // 32) + 1
            
            for i in range(n_batches):
                batch_size = min(32, n_needed - i*32)
                if batch_size <= 0:
                    break
                
                noise = np.random.normal(0, 1, (batch_size, latent_dim))
                generated_ecg = generator.predict(noise, verbose=0)
                generated_ecg += 0.02 * np.random.randn(*generated_ecg.shape)
                
                generated_samples.append(generated_ecg)
            
            if generated_samples:
                generated_samples = np.vstack(generated_samples)[:n_needed]
                X_augmented.append(generated_samples)
                y_augmented.append(np.array([class_idx] * len(generated_samples)))
        else:
            print(f" -> already balanced")
    
    X_augmented = np.vstack(X_augmented)
    y_augmented = np.hstack(y_augmented)
    
    idx = np.random.permutation(len(X_augmented))
    X_augmented = X_augmented[idx]
    y_augmented = y_augmented[idx]
    
    print(f"\n✅ Augmentation complete:")
    print(f"   Original: {len(X_train)} samples")
    print(f"   Augmented: {len(X_augmented)} samples")
    
    return X_augmented, y_augmented

# ==========================================
# CALCULATE CLASS METRICS
# ==========================================
def calculate_class_metrics(y_true, y_pred, num_classes):
    """Calculate per-class metrics including specificity"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    precision = []
    recall = []
    specificity = []
    f1 = []
    support = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        s = tn / (tn + fp) if (tn + fp) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        precision.append(p)
        recall.append(r)
        specificity.append(s)
        f1.append(f)
        support.append(cm[i, :].sum())
    
    return {
        'p': np.array(precision),
        'r': np.array(recall),
        's': np.array(specificity),
        'f1': np.array(f1),
        'support': np.array(support)
    }

# ==========================================
# TRAIN WITH GAN AUGMENTATION
# ==========================================
def train_with_gan_augmentation(X_train, y_train, X_val, y_val, input_shape, num_classes):
    """Train dengan GAN augmentation"""
    
    print(f"\n⚡ Training with GAN augmentation...")
    
    # Build and train GAN
    generator, discriminator = build_ecg_gan(input_shape)
    
    print(f"\n📐 Model Shapes:")
    print(f"   Input shape: {input_shape}")
    print(f"   Generator output: {generator.output_shape}")
    
    trained_generator = train_ecg_gan(generator, discriminator, X_train, epochs=30, batch_size=BATCH_SIZE)
    
    # Augment data
    X_train_aug, y_train_aug = augment_with_gan(trained_generator, X_train, y_train, n_samples_per_class=800)
    
    # Cleanup GAN models
    del generator, discriminator, trained_generator
    gc.collect()
    
    # Build classification model
    model = build_cnn_bilstm_attention_model(input_shape, num_classes)
    
    # Compile
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Class weights
    try:
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train_aug),
            y=y_train_aug
        )
        class_weights = dict(enumerate(class_weights))
    except:
        class_weights = None
    
    # Train
    print(f"\n🏋️ Training CNN-BiLSTM-Attention model...")
    print(f"   Batch Size: {BATCH_SIZE}")
    start_time = time.time()
    
    history = model.fit(
        X_train_aug, to_categorical(y_train_aug, num_classes),
        validation_data=(X_val, to_categorical(y_val, num_classes)),
        epochs=50,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    train_time = time.time() - start_time
    
    # Inference time
    inf_start = time.time()
    _ = model.predict(X_val, verbose=0)
    inf_time = time.time() - inf_start
    
    # Evaluate
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    # Calculate class metrics
    class_metrics = calculate_class_metrics(y_val, y_pred, num_classes)
    
    print(f"\n✅ Training completed in {train_time:.1f}s")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    return model, history, acc, f1, train_time, inf_time, class_metrics

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y_raw, test_size=0.2, random_state=SEED, stratify=y_raw
        )
        
        # Standardize
        print(f"\n📊 Preprocessing data...")
        scaler = StandardScaler()
        X_train_2d = X_train.reshape(-1, X_train.shape[-1])
        X_test_2d = X_test.reshape(-1, X_test.shape[-1])
        
        X_train_scaled_2d = scaler.fit_transform(X_train_2d)
        X_test_scaled_2d = scaler.transform(X_test_2d)
        
        X_train_scaled = X_train_scaled_2d.reshape(X_train.shape)
        X_test_scaled = X_test_scaled_2d.reshape(X_test.shape)
        
        print(f"   Train: {X_train_scaled.shape}")
        print(f"   Test: {X_test_scaled.shape}")
        
        # Train with GAN
        model, history, acc, f1, train_time, inf_time, class_metrics = train_with_gan_augmentation(
            X_train_scaled, y_train,
            X_test_scaled, y_test,
            input_shape=X_train_scaled.shape[1:],
            num_classes=NUM_CLASSES
        )
        
        # Save model
        model.save('best_model_bilstm_gan.h5')
        print(f"\n💾 Model saved: best_model_bilstm_gan.h5")
        
        # ==========================================
        # GENERATE TABLES
        # ==========================================
        print("\n" + "="*80)
        print("📊 GENERATING TABLES")
        print("="*80)
        
        # Tabel 1: Evaluasi BiLSTM-GAN
        fold_results = [{
            'Akurasi': acc,
            'F1_Score': f1,
            'Train_Time': train_time,
            'Inf_Time': inf_time
        }]
        
        df_res = pd.DataFrame(fold_results)
        
        t1 = pd.DataFrame({
            'Metric': ['Rata-rata Akurasi', 'Akurasi Terbaik', 'Rata-rata F1'],
            'Value': [df_res['Akurasi'].mean(), acc, df_res['F1_Score'].mean()]
        })
        t1.to_csv("Tabel_1_Evaluasi_BiLSTM_GAN.csv", index=False)
        print("✅ Tabel 1 Saved: Tabel_1_Evaluasi_BiLSTM_GAN.csv")
        
        # Tabel 2: Detail Kelas BiLSTM-GAN
        t2 = pd.DataFrame({
            'Kelas': class_names,
            'Recall': class_metrics['r'],
            'Spesifisitas': class_metrics['s'],
            'Presisi': class_metrics['p'],
            'F1': class_metrics['f1']
        })
        t2.to_csv("Tabel_2_Detail_Kelas_BiLSTM_GAN.csv", index=False)
        print("✅ Tabel 2 Saved: Tabel_2_Detail_Kelas_BiLSTM_GAN.csv")
        
        # Tabel 3: Efisiensi BiLSTM-GAN
        model_size = os.path.getsize("best_model_bilstm_gan.h5") / (1024*1024)
        avg_inf_time_ms = (inf_time / len(X_test_scaled)) * 1000
        
        t3 = pd.DataFrame({
            'Metric': ['Total Parameter', 'Size (MB)', 'Avg Train Time (s)', 'Avg Inf Time (ms)'],
            'Value': [model.count_params(), model_size, train_time, avg_inf_time_ms]
        })
        t3.to_csv("Tabel_3_Efisiensi_BiLSTM_GAN.csv", index=False)
        print("✅ Tabel 3 Saved: Tabel_3_Efisiensi_BiLSTM_GAN.csv")
        
        # Tabel 4: Konsistensi BiLSTM-GAN
        t4 = pd.DataFrame({
            'Metric': ['Avg Accuracy', 'Std Dev', 'Min', 'Max'],
            'Value': [df_res['Akurasi'].mean(), df_res['Akurasi'].std(), df_res['Akurasi'].min(), df_res['Akurasi'].max()]
        })
        t4.to_csv("Tabel_4_Konsistensi_BiLSTM_GAN.csv", index=False)
        print("✅ Tabel 4 Saved: Tabel_4_Konsistensi_BiLSTM_GAN.csv")
        
        # ==========================================
        # GENERATE GRAPHS
        # ==========================================
        print("\n" + "="*80)
        print("📈 GENERATING GRAPHS")
        print("="*80)
        
        # Plot 1: Learning Curve
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Val', linewidth=2)
        plt.title('Accuracy BiLSTM-GAN', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train', linewidth=2)
        plt.plot(history.history['val_loss'], label='Val', linewidth=2)
        plt.title('Loss BiLSTM-GAN', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Plot_1_Learning_Curve_BiLSTM_GAN.png', dpi=300, bbox_inches='tight')
        print("✅ Plot 1 Saved: Plot_1_Learning_Curve_BiLSTM_GAN.png")
        plt.close()
        
        # Plot 2: Confusion Matrix
        y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix BiLSTM-GAN', fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('Plot_2_CM_BiLSTM_GAN.png', dpi=300, bbox_inches='tight')
        print("✅ Plot 2 Saved: Plot_2_CM_BiLSTM_GAN.png")
        plt.close()
        
        # Plot 3: ROC Curve
        y_probs = model.predict(X_test_scaled, verbose=0)
        y_test_bin = label_binarize(y_test, classes=range(NUM_CLASSES))
        
        plt.figure(figsize=(8, 6))
        for i in range(NUM_CLASSES):
            if np.sum(y_test_bin[:, i]) > 0:
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC={roc_auc:.2f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve BiLSTM-GAN', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('Plot_3_ROC_BiLSTM_GAN.png', dpi=300, bbox_inches='tight')
        print("✅ Plot 3 Saved: Plot_3_ROC_BiLSTM_GAN.png")
        plt.close()
        
        # Plot 4: Attention Map
        try:
            att_model = Model(inputs=model.input, outputs=model.get_layer('attention_weights').output)
            att_w = att_model.predict(X_test_scaled[0:1], verbose=0)
            
            plt.figure(figsize=(10, 3))
            plt.plot(att_w[0].flatten(), color='purple', linewidth=2)
            plt.title('Attention Map BiLSTM-GAN', fontweight='bold')
            plt.xlabel('Time Step')
            plt.ylabel('Attention Weight')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('Plot_4_Attention_BiLSTM_GAN.png', dpi=300, bbox_inches='tight')
            print("✅ Plot 4 Saved: Plot_4_Attention_BiLSTM_GAN.png")
            plt.close()
        except Exception as e:
            print(f"⚠️  Plot 4 (Attention Map) skipped: {e}")
        
        # Plot 5: t-SNE
        try:
            inter_model = Model(inputs=model.input, outputs=model.get_layer('embedding').output)
            feats = inter_model.predict(X_test_scaled, verbose=0)
            
            print("   Computing t-SNE (this may take a moment)...")
            # Perplexity must be less than n_samples
            n_samples_tsne = len(X_test_scaled)
            perplexity_val = min(30, max(5, (n_samples_tsne - 1) // 3))
            
            emb = TSNE(n_components=2, random_state=42, perplexity=perplexity_val).fit_transform(feats)
            
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(emb[:, 0], emb[:, 1], 
                                c=y_test, cmap='viridis', 
                                alpha=0.6, s=50)
            plt.colorbar(scatter, ticks=range(NUM_CLASSES), label='Class')
            plt.title('t-SNE BiLSTM-GAN', fontweight='bold')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('Plot_5_TSNE_BiLSTM_GAN.png', dpi=300, bbox_inches='tight')
            print("✅ Plot 5 Saved: Plot_5_TSNE_BiLSTM_GAN.png")
            plt.close()
        except Exception as e:
            print(f"⚠️  Plot 5 (t-SNE) skipped: {e}")
        
        # ==========================================
        # FINAL SUMMARY
        # ==========================================
        print("\n" + "="*80)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print(f"\n🏆 FINAL RESULTS:")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Training Time: {train_time:.1f}s")
        print(f"   Inference Time: {avg_inf_time_ms:.2f}ms per sample")
        
        print(f"\n🔧 ARCHITECTURE:")
        print(f"   ✅ CNN Feature Extractor (3 layers)")
        print(f"   ✅ Bidirectional LSTM")
        print(f"   ✅ Attention Mechanism")
        print(f"   ✅ ECG GAN for data balancing")
        print(f"   ✅ GPU Acceleration")
        
        print(f"\n📁 OUTPUT FILES:")
        print("\n   TABLES:")
        print("   1. Tabel_1_Evaluasi_BiLSTM_GAN.csv")
        print("   2. Tabel_2_Detail_Kelas_BiLSTM_GAN.csv")
        print("   3. Tabel_3_Efisiensi_BiLSTM_GAN.csv")
        print("   4. Tabel_4_Konsistensi_BiLSTM_GAN.csv")
        
        print("\n   PLOTS:")
        print("   1. Plot_1_Learning_Curve_BiLSTM_GAN.png")
        print("   2. Plot_2_CM_BiLSTM_GAN.png")
        print("   3. Plot_3_ROC_BiLSTM_GAN.png")
        print("   4. Plot_4_Attention_BiLSTM_GAN.png")
        print("   5. Plot_5_TSNE_BiLSTM_GAN.png")
        
        print("\n   MODEL:")
        print("   • best_model_bilstm_gan.h5")
        
        print("\n✅ BiLSTM-GAN SELESAI. Semua file PNG dan CSV telah siap.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        gc.collect()
        print(f"\n🏁 PROGRAM FINISHED")
