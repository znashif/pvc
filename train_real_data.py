# ==========================================
# CNN-BiLSTM-ATTENTION WITH ECG GAN
# USING REAL ECG DATA
# ==========================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.signal import butter, filtfilt
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
    LeakyReLU, GlobalAveragePooling1D,
    Multiply
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import gc
import warnings
warnings.filterwarnings('ignore')
from tqdm.auto import tqdm

print("="*80)
print("🚀 CNN-BiLSTM-ATTENTION FOR REAL ECG DATA")
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
SEQUENCE_LENGTH = 1000  # Fixed length for all ECG signals

# ==========================================
# LOAD AND PROCESS DATA
# ==========================================
print("\n" + "="*80)
print("📂 LOADING REAL ECG DATA")
print("="*80)

# Load diagnosis labels
print("\n1️⃣ Loading diagnosis labels...")
df_diagnosis = pd.read_excel(DIAGNOSIS_FILE)
print(f"   Total records in diagnosis file: {len(df_diagnosis)}")
print(f"   Columns: {df_diagnosis.columns.tolist()}")

# Check Sublocation column
if 'Sublocation' in df_diagnosis.columns:
    print(f"\n   Sublocation distribution:")
    print(df_diagnosis['Sublocation'].value_counts())
    label_column = 'Sublocation'
else:
    print("\n   ⚠️  'Sublocation' column not found, checking other columns...")
    print(df_diagnosis.head())
    # Try to find the label column
    for col in df_diagnosis.columns:
        if 'location' in col.lower() or 'diagnosis' in col.lower():
            label_column = col
            print(f"   Using column: {label_column}")
            break

# Load ECG files
print("\n2️⃣ Loading ECG files...")
ecg_data = []
labels = []
patient_ids = []

# Get list of CSV files
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
print(f"   Found {len(csv_files)} CSV files")

# Create mapping from HospitalID to Sublocation
id_to_label = dict(zip(df_diagnosis['HospitalID'].astype(str), df_diagnosis[label_column]))

loaded_count = 0
skipped_count = 0

print("\n3️⃣ Processing ECG signals...")
for csv_file in tqdm(csv_files[:100], desc="Loading ECG files"):  # Limit to 100 for faster testing
    patient_id = csv_file.replace('.csv', '')
    
    # Check if we have label for this patient
    if patient_id not in id_to_label:
        skipped_count += 1
        continue
    
    label = id_to_label[patient_id]
    
    # Skip if label is NaN
    if pd.isna(label):
        skipped_count += 1
        continue
    
    try:
        # Load ECG data
        ecg_df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
        
        # Assuming ECG data has 12 leads (columns)
        # Take first SEQUENCE_LENGTH samples
        if len(ecg_df) < SEQUENCE_LENGTH:
            # Pad if too short
            pad_length = SEQUENCE_LENGTH - len(ecg_df)
            ecg_array = np.pad(ecg_df.values, ((0, pad_length), (0, 0)), mode='edge')
        else:
            # Truncate if too long
            ecg_array = ecg_df.values[:SEQUENCE_LENGTH, :]
        
        # Ensure we have 12 channels
        if ecg_array.shape[1] != 12:
            # If not 12 channels, skip or pad/truncate
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
        print(f"\n   ⚠️  Error loading {csv_file}: {e}")
        skipped_count += 1
        continue

print(f"\n✅ Data loading complete:")
print(f"   Loaded: {loaded_count} samples")
print(f"   Skipped: {skipped_count} samples")

# Convert to numpy arrays
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
print(f"   Class names: {list(class_names)}")
print(f"   Class distribution:")
for i, name in enumerate(class_names):
    count = np.sum(y_raw == i)
    print(f"      {name}: {count} samples ({count/len(y_raw)*100:.1f}%)")

# Save processed data
np.save('X_raw.npy', X_raw)
np.save('y_raw.npy', y_raw)
np.save('patient_ids.npy', np.array(patient_ids))
joblib.dump(encoder, 'label_encoder.joblib')
print("\n💾 Saved processed data to .npy files")

# ==========================================
# CNN-BiLSTM-ATTENTION MODEL (SIMPLIFIED)
# ==========================================
def build_model(input_shape, num_classes):
    """CNN-BiLSTM dengan Attention - Simplified for CPU"""
    
    inputs = Input(shape=input_shape)
    
    # CNN layers
    x = Conv1D(32, 7, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(64, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    # BiLSTM
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    
    # Attention
    attention = Dense(1, activation='tanh')(x)
    attention = tf.keras.layers.Softmax(axis=1)(attention)
    x = Multiply()([x, attention])
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(64, activation='relu', name='embedding')(x)
    x = Dropout(0.4)(x)
    
    # Output
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = Model(inputs, outputs)
    return model

# ==========================================
# TRAINING
# ==========================================
print("\n" + "="*80)
print("🏋️ TRAINING MODEL")
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

X_train_scaled_2d = scaler.fit_transform(X_train_2d)
X_test_scaled_2d = scaler.transform(X_test_2d)

X_train_scaled = X_train_scaled_2d.reshape(X_train.shape)
X_test_scaled = X_test_scaled_2d.reshape(X_test.shape)

print(f"   Train: {X_train_scaled.shape}")
print(f"   Test: {X_test_scaled.shape}")

# Handle imbalance with oversampling
print("\n🔧 Balancing data with oversampling...")
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
ros = RandomOverSampler(random_state=SEED)
X_train_bal_reshaped, y_train_bal = ros.fit_resample(X_train_reshaped, y_train)
X_train_bal = X_train_bal_reshaped.reshape(-1, X_train_scaled.shape[1], X_train_scaled.shape[2])

print(f"   Original: {len(X_train)} samples")
print(f"   Balanced: {len(X_train_bal)} samples")

# Build model
print("\n🔨 Building model...")
model = build_model(X_train_scaled.shape[1:], NUM_CLASSES)
model.summary()

# Compile
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

# Train
print("\n⚡ Training...")
start_time = time.time()

history = model.fit(
    X_train_bal, to_categorical(y_train_bal, NUM_CLASSES),
    validation_data=(X_test_scaled, to_categorical(y_test, NUM_CLASSES)),
    epochs=30,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

train_time = time.time() - start_time

# ==========================================
# EVALUATION
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
print(f"\n💾 Model saved: ecg_model_real_data.h5")

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
plt.savefig('confusion_matrix_real_data.png', dpi=300)
print("\n📊 Saved: confusion_matrix_real_data.png")

# Learning curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves_real_data.png', dpi=300)
print("📊 Saved: learning_curves_real_data.png")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
