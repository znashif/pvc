# ==========================================
# CNN-BiLSTM-ATTENTION WITH ECG GAN
# USING REAL DATA FROM PVCVTECGData
# ==========================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 

import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, recall_score, precision_score, 
    roc_curve, auc, roc_auc_score, matthews_corrcoef
)
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, BatchNormalization,
    Dense, Dropout, LSTM, Bidirectional,
    LeakyReLU, GlobalAveragePooling1D, Multiply,
    Reshape, Conv1DTranspose
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import gc
import warnings
warnings.filterwarnings('ignore')
from tqdm.auto import tqdm

print("="*80)
print("🚀 CNN-BiLSTM-ATTENTION + ECG GAN - REAL ECG DATA")
print("="*80)
print(f"TensorFlow Version: {tf.__version__}")

# ==========================================
# CONFIGURATION
# ==========================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = 'D:/GAN/PVCVTECGData'
DIAGNOSIS_FILE = 'D:/GAN/Diagnosis.xlsx'
BATCH_SIZE = 32
SEQUENCE_LENGTH = 1000
MAX_FILES = 200  # Limit files untuk testing, set None untuk semua

# ==========================================
# LOAD REAL DATA
# ==========================================
print("\n" + "="*80)
print("📂 LOADING REAL ECG DATA")
print("="*80)

# 1. Load diagnosis labels
print("\n1️⃣ Loading diagnosis labels...")
df_diagnosis = pd.read_excel(DIAGNOSIS_FILE)
print(f"   Total records: {len(df_diagnosis)}")
print(f"   Columns: {df_diagnosis.columns.tolist()}")

# Find label column
label_column = 'Sublocation' if 'Sublocation' in df_diagnosis.columns else None
if label_column:
    print(f"\n   Using label column: {label_column}")
    print(f"   Label distribution:")
    print(df_diagnosis[label_column].value_counts())
else:
    print("\n   ERROR: Sublocation column not found!")
    print(df_diagnosis.head())
    exit(1)

# 2. Load ECG files
print("\n2️⃣ Loading ECG files...")
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
print(f"   Found {len(csv_files)} CSV files")

if MAX_FILES:
    csv_files = csv_files[:MAX_FILES]
    print(f"   Processing first {MAX_FILES} files for testing")

# Create mapping
id_to_label = dict(zip(df_diagnosis['HospitalID'].astype(str), df_diagnosis[label_column]))

# Load data
ecg_data = []
labels = []
patient_ids = []

loaded_count = 0
skipped_count = 0

print("\n3️⃣ Processing ECG signals...")
for csv_file in tqdm(csv_files, desc="Loading"):
    patient_id = csv_file.replace('.csv', '')
    
    if patient_id not in id_to_label:
        skipped_count += 1
        continue
    
    label = id_to_label[patient_id]
    if pd.isna(label):
        skipped_count += 1
        continue
    
    try:
        ecg_df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
        
        # Handle sequence length
        if len(ecg_df) < SEQUENCE_LENGTH:
            pad_length = SEQUENCE_LENGTH - len(ecg_df)
            ecg_array = np.pad(ecg_df.values, ((0, pad_length), (0, 0)), mode='edge')
        else:
            ecg_array = ecg_df.values[:SEQUENCE_LENGTH, :]
        
        # Ensure 12 channels
        if ecg_array.shape[1] != 12:
            if ecg_array.shape[1] < 12:
                pad_channels = 12 - ecg_array.shape[1]
                ecg_array = np.pad(ecg_array, ((0, 0), (0, pad_channels)), mode='edge')
            else:
                ecg_array = ecg_array[:, :12]
        
        ecg_data.append(ecg_array.astype(np.float32))
        labels.append(label)
        patient_ids.append(patient_id)
        loaded_count += 1
        
    except Exception as e:
        skipped_count += 1
        continue

print(f"\n✅ Data loading complete:")
print(f"   Loaded: {loaded_count} samples")
print(f"   Skipped: {skipped_count} samples")

if loaded_count < 20:
    print("\n❌ ERROR: Not enough samples loaded!")
    print("   Please check if data files exist and labels match.")
    exit(1)

# Convert to arrays
X_raw = np.array(ecg_data)
y_raw_labels = np.array(labels)

# Encode labels
encoder = LabelEncoder()
y_raw = encoder.fit_transform(y_raw_labels)
class_names = encoder.classes_
NUM_CLASSES = len(class_names)

print(f"\n📊 Dataset Summary:")
print(f"   Shape: {X_raw.shape}")
print(f"   Classes: {NUM_CLASSES}")
for i, name in enumerate(class_names):
    count = np.sum(y_raw == i)
    print(f"   {name}: {count} samples ({count/len(y_raw)*100:.1f}%)")

# Save processed data
np.save('X_raw.npy', X_raw)
np.save('y_raw.npy', y_raw)
np.save('patient_ids.npy', np.array(patient_ids))
joblib.dump(encoder, 'label_encoder.joblib')
print("\n💾 Saved processed data")

# ==========================================
# MODEL ARCHITECTURES
# ==========================================

def build_cnn_bilstm_attention_model(input_shape, num_classes):
    """CNN-BiLSTM dengan Attention"""
    inputs = Input(shape=input_shape)
    
    # CNN
    x = Conv1D(32, 7, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(64, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    # BiLSTM
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    
    # Attention
    attention = Dense(1, activation='tanh', name='attention_dense')(x)
    attention = tf.keras.layers.Softmax(axis=1, name='attention_weights')(attention)
    x = Multiply()([x, attention])
    
    # Pooling
    x = GlobalAveragePooling1D()(x)
    
    # Dense
    x = Dense(128, activation='relu', name='embedding')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    return Model(inputs, outputs, name='CNN_BiLSTM_Attention')

def build_ecg_gan(input_shape, latent_dim=100):
    """ECG GAN - Simplified"""
    seq_len, n_channels = input_shape
    initial_len = seq_len // 16
    
    # Generator
    gen_input = Input(shape=(latent_dim,))
    x = Dense(64 * initial_len)(gen_input)
    x = LeakyReLU(0.2)(x)
    x = Reshape((initial_len, 64))(x)
    
    x = Conv1DTranspose(32, 5, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv1DTranspose(16, 5, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv1DTranspose(8, 5, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv1DTranspose(n_channels, 5, strides=2, padding='same', activation='tanh')(x)
    
    generator = Model(gen_input, x, name='Generator')
    
    # Discriminator
    disc_input = Input(shape=input_shape)
    x = Conv1D(32, 5, padding='same')(disc_input)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(64, 5, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(128, 3, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = GlobalAveragePooling1D()(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid', dtype='float32')(x)
    
    discriminator = Model(disc_input, outputs, name='Discriminator')
    
    return generator, discriminator

# ==========================================
# TRAINING
# ==========================================
print("\n" + "="*80)
print("🏋️ TRAINING")
print("="*80)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=SEED, stratify=y_raw
)

# Standardize
print("\n📊 Preprocessing...")
scaler = StandardScaler()
X_train_2d = X_train.reshape(-1, X_train.shape[-1])
X_test_2d = X_test.reshape(-1, X_test.shape[-1])

X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape)

print(f"   Train: {X_train_scaled.shape}")
print(f"   Test: {X_test_scaled.shape}")

# Balance with oversampling (simpler than GAN for now)
print("\n🔧 Balancing data...")
X_train_2d = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
ros = RandomOverSampler(random_state=SEED)
X_train_bal, y_train_bal = ros.fit_resample(X_train_2d, y_train)
X_train_bal = X_train_bal.reshape(-1, X_train_scaled.shape[1], X_train_scaled.shape[2])

print(f"   Original: {len(X_train)}")
print(f"   Balanced: {len(X_train_bal)}")

# Build model
print("\n🔨 Building model...")
model = build_cnn_bilstm_attention_model(X_train_scaled.shape[1:], NUM_CLASSES)

model.compile(
    optimizer=Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
print("\n⚡ Training...")
start_time = time.time()

history = model.fit(
    X_train_bal, to_categorical(y_train_bal, NUM_CLASSES),
    validation_data=(X_test_scaled, to_categorical(y_test, NUM_CLASSES)),
    epochs=50,
    batch_size=BATCH_SIZE,
    callbacks=[
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
    ],
    verbose=1
)

train_time = time.time() - start_time

# ==========================================
# EVALUATION & OUTPUTS
# ==========================================
print("\n" + "="*80)
print("📊 EVALUATION")
print("="*80)

y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n🏆 Results:")
print(f"   Accuracy: {acc:.4f}")
print(f"   F1-Score: {f1:.4f}")
print(f"   Training Time: {train_time:.1f}s")

# Save model
model.save('ecg_model_real_data.h5')
print(f"\n💾 Model saved")

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Real ECG Data')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_real.png', dpi=300)
print("\n📊 Saved: confusion_matrix_real.png")

# Learning curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves_real.png', dpi=300)
print("📊 Saved: learning_curves_real.png")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE WITH REAL DATA!")
print("="*80)
print(f"\n📁 Output files:")
print("   • ecg_model_real_data.h5")
print("   • confusion_matrix_real.png")
print("   • learning_curves_real.png")
print("   • X_raw.npy, y_raw.npy, patient_ids.npy")
print("   • label_encoder.joblib")
