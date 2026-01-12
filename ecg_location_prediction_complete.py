"""
ECG 12-Lead Location Prediction using CNN-BiLSTM + Attention and ECG GAN
Dataset: PVCVTECGData with Diagnosis.xlsx labels
Task: Predict location origin from 12-lead ECG data (4 main classes)
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Sklearn utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# GPU Configuration
print("=" * 80)
print("GPU Configuration")
print("=" * 80)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ {len(gpus)} GPU(s) available and configured")
    except RuntimeError as e:
        print(f"✗ GPU configuration error: {e}")
else:
    print("✗ No GPU available, using CPU")

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

class ECGDataLoader:
    """Load and preprocess ECG data from CSV files"""
    
    def __init__(self, data_dir, label_file, target_length=5000):
        """
        Args:
            data_dir: Directory containing ECG CSV files
            label_file: Excel file with patient labels
            target_length: Target sequence length for ECG signals
        """
        self.data_dir = data_dir
        self.label_file = label_file
        self.target_length = target_length
        self.label_encoder = LabelEncoder()
        
    def load_labels(self):
        """Load and process labels from Excel file"""
        print("\n" + "=" * 80)
        print("Loading Labels")
        print("=" * 80)
        
        # Read Excel file
        df_labels = pd.read_excel(self.label_file)
        print(f"✓ Loaded {len(df_labels)} patient records")
        print(f"✓ Columns: {list(df_labels.columns)}")
        
        # Display first few rows
        print("\nFirst 5 rows:")
        print(df_labels.head())
        
        # Map to 4 main classes (adjust based on your actual label column)
        # This is a placeholder - adjust according to your actual data
        print("\nLabel distribution:")
        if 'Diagnosis' in df_labels.columns:
            print(df_labels['Diagnosis'].value_counts())
        
        return df_labels
    
    def load_ecg_file(self, filepath, segment_by_beat=True, visualize_first=False):
        """
        Load single ECG CSV file
        
        Args:
            filepath: Path to ECG file
            segment_by_beat: If True, segment into individual heartbeats
            visualize_first: If True, visualize first file's segmentation
        
        Returns:
            List of ECG segments (or list with single whole recording)
        """
        try:
            # Read CSV file
            ecg_data = pd.read_csv(filepath)
            
            # Extract 12-lead data
            leads = ['aVF', 'aVL', 'aVR', 'I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            
            if all(lead in ecg_data.columns for lead in leads):
                ecg_array = ecg_data[leads].values
            else:
                # If columns don't match, use all columns
                ecg_array = ecg_data.values
            
            # Normalize each lead
            ecg_array = self.normalize_signal(ecg_array)
            
            if segment_by_beat:
                # Segment into individual heartbeats
                segments = self.segment_ecg_by_heartbeat(
                    ecg_array, 
                    sampling_rate=500,
                    visualize=visualize_first
                )
                
                if len(segments) == 0:
                    # If no segments found, fall back to whole recording
                    print(f"  Warning: No heartbeats detected in {filepath}, using whole recording")
                    ecg_array = self.resize_signal(ecg_array, self.target_length)
                    return [ecg_array]
                
                return segments
            else:
                # Use whole recording (old approach)
                ecg_array = self.resize_signal(ecg_array, self.target_length)
                return [ecg_array]
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return []

    
    def normalize_signal(self, signal):
        """Normalize ECG signal using z-score normalization"""
        mean = np.mean(signal, axis=0)
        std = np.std(signal, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        return (signal - mean) / std
    
    def resize_signal(self, signal, target_length):
        """Resize signal to target length using interpolation"""
        current_length = signal.shape[0]
        
        if current_length == target_length:
            return signal
        
        # Linear interpolation
        indices = np.linspace(0, current_length - 1, target_length)
        resized_signal = np.zeros((target_length, signal.shape[1]))
        
        for i in range(signal.shape[1]):
            resized_signal[:, i] = np.interp(indices, np.arange(current_length), signal[:, i])
        
        return resized_signal
    
    def segment_ecg_by_heartbeat(self, signal, sampling_rate=500, visualize=False):
        """
        Segment ECG into individual heartbeats using R-peak detection
        
        Args:
            signal: ECG signal (timesteps, 12_leads)
            sampling_rate: Sampling rate in Hz (default 500)
            visualize: If True, plot example segmentation
        
        Returns:
            List of heartbeat segments
        """
        from scipy.signal import find_peaks
        
        # Use Lead II for R-peak detection (usually clearest)
        lead_ii = signal[:, 3]  # Lead II is at index 3
        
        # Normalize for peak detection
        lead_ii_norm = (lead_ii - np.mean(lead_ii)) / (np.std(lead_ii) + 1e-8)
        
        # Find R-peaks
        # Min distance = 0.6s (100 bpm max) = 300 samples at 500Hz
        min_distance = int(0.6 * sampling_rate)
        
        # Height threshold: peaks should be above mean + 0.5*std
        height_threshold = 0.5
        
        peaks, properties = find_peaks(
            lead_ii_norm, 
            distance=min_distance,
            height=height_threshold,
            prominence=0.3
        )
        
        # Define window around R-peak
        # Standard: 200ms before R-peak, 400ms after R-peak = 600ms total
        window_before = int(0.2 * sampling_rate)  # 100 samples at 500Hz
        window_after = int(0.4 * sampling_rate)   # 200 samples at 500Hz
        total_window = window_before + window_after
        
        segments = []
        valid_peaks = []
        
        for peak in peaks:
            start = peak - window_before
            end = peak + window_after
            
            # Check if segment is within bounds
            if start >= 0 and end <= len(signal):
                segment = signal[start:end, :]
                
                # Verify segment has correct length
                if segment.shape[0] == total_window:
                    segments.append(segment)
                    valid_peaks.append(peak)
        
        # Visualize if requested
        if visualize and len(segments) > 0:
            self.visualize_segmentation(signal, valid_peaks, segments[0], 
                                       window_before, window_after, sampling_rate)
        
        return segments
    
    def visualize_segmentation(self, full_signal, peaks, example_segment, 
                              window_before, window_after, sampling_rate):
        """Visualize ECG segmentation with example heartbeat"""
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        
        # Plot 1: Full ECG with detected R-peaks
        ax1 = axes[0]
        time_full = np.arange(len(full_signal)) / sampling_rate
        
        # Plot all 12 leads (stacked for visibility)
        lead_names = ['aVF', 'aVL', 'aVR', 'I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        offset = 0
        for i in range(12):
            ax1.plot(time_full, full_signal[:, i] + offset, alpha=0.7, linewidth=0.8, label=lead_names[i])
            offset += 3
        
        # Mark R-peaks
        for peak in peaks:
            ax1.axvline(x=peak/sampling_rate, color='red', alpha=0.5, linestyle='--', linewidth=1)
        
        ax1.set_title('Full ECG Recording with Detected R-Peaks (Red Lines)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Amplitude (offset for visibility)', fontsize=12)
        ax1.legend(loc='upper right', ncol=6, fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Zoomed view around first few heartbeats
        ax2 = axes[1]
        zoom_samples = min(2000, len(full_signal))  # Show ~4 seconds
        time_zoom = np.arange(zoom_samples) / sampling_rate
        
        # Plot Lead II (clearest for heartbeats)
        ax2.plot(time_zoom, full_signal[:zoom_samples, 3], linewidth=2, color='blue', label='Lead II')
        
        # Mark R-peaks in zoomed view
        for peak in peaks:
            if peak < zoom_samples:
                ax2.axvline(x=peak/sampling_rate, color='red', alpha=0.7, linestyle='--', linewidth=2)
                ax2.plot(peak/sampling_rate, full_signal[peak, 3], 'ro', markersize=10)
        
        ax2.set_title('Zoomed View: Lead II with R-Peak Detection', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Amplitude', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Example single heartbeat (all 12 leads)
        ax3 = axes[2]
        time_beat = np.arange(len(example_segment)) / sampling_rate * 1000  # Convert to ms
        
        for i in range(12):
            ax3.plot(time_beat, example_segment[:, i], linewidth=1.5, label=lead_names[i], alpha=0.8)
        
        # Mark R-peak position (at window_before)
        r_peak_time = window_before / sampling_rate * 1000
        ax3.axvline(x=r_peak_time, color='red', linestyle='--', linewidth=2, label='R-Peak')
        
        ax3.set_title(f'Example Single Heartbeat (600ms window: 200ms before + 400ms after R-peak)', 
                     fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time (milliseconds)', fontsize=12)
        ax3.set_ylabel('Amplitude', fontsize=12)
        ax3.legend(loc='upper right', ncol=4, fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ECG_Segmentation_Visualization.png', dpi=300, bbox_inches='tight')
        print("\n✓ Segmentation visualization saved as 'ECG_Segmentation_Visualization.png'")
        plt.close()

    
    def create_4_class_mapping(self, df_labels, diagnosis_column='Diagnosis'):
        """
        Map original diagnoses to 4 main classes based on cardiac regions
        Data format: LeftRight column + Sublocation column
        """
        # Correct mapping based on user's specification
        class_mapping = {
            # Left Ventricular Region
            "Left_AMC": "Left_Ventricular_Region",
            "Left_LCC": "Left_Ventricular_Region", 
            "Left_RCC": "Left_Ventricular_Region",
            "Left_Summit": "Left_Ventricular_Region",
            
            # Right Ventricular Region
            "Right_LC": "Right_Ventricular_Region",
            "Right_RC": "Right_Ventricular_Region", 
            "Right_AC": "Right_Ventricular_Region",
            "Right_FreeWall": "Right_Ventricular_Region",
            
            # Outflow Tract Region
            "Right_RVOTOther": "Outflow_Tract_Region",
            "Left_LCC-RCC Ommisure": "Outflow_Tract_Region",
            
            # Septal Region
            "Right_PosteriorSeptal": "Septal_Region",
            "Right_AnteriorSeptal": "Septal_Region"
        }
        
        # Check if we have LeftRight and Sublocation columns (actual data format)
        if 'LeftRight' in df_labels.columns and 'Sublocation' in df_labels.columns:
            print("✓ Using LeftRight + Sublocation columns")
            # Combine LeftRight and Sublocation to create full location label
            df_labels['Full_Location'] = df_labels['LeftRight'] + '_' + df_labels['Sublocation']
            df_labels['Location_Class'] = df_labels['Full_Location'].map(class_mapping)
            
            # Check for unmapped values
            unmapped = df_labels[df_labels['Location_Class'].isna()]
            if len(unmapped) > 0:
                print(f"⚠ Warning: {len(unmapped)} records with unmapped diagnoses")
                print(f"  Unmapped values: {unmapped['Full_Location'].unique()}")
                # Remove unmapped records
                df_labels = df_labels.dropna(subset=['Location_Class'])
            
        elif diagnosis_column in df_labels.columns:
            df_labels['Location_Class'] = df_labels[diagnosis_column].map(class_mapping)
            
            # Check for unmapped values
            unmapped = df_labels[df_labels['Location_Class'].isna()]
            if len(unmapped) > 0:
                print(f"⚠ Warning: {len(unmapped)} records with unmapped diagnoses")
                print(f"  Unmapped values: {unmapped[diagnosis_column].unique()}")
                # Remove unmapped records
                df_labels = df_labels.dropna(subset=['Location_Class'])
        else:
            # Try to find the diagnosis column
            possible_columns = [col for col in df_labels.columns if 'diag' in col.lower() or 'location' in col.lower()]
            if possible_columns:
                print(f"⚠ Using column '{possible_columns[0]}' as diagnosis column")
                df_labels['Location_Class'] = df_labels[possible_columns[0]].map(class_mapping)
                df_labels = df_labels.dropna(subset=['Location_Class'])
            else:
                raise ValueError(f"No diagnosis column found. Available columns: {list(df_labels.columns)}")
        
        return df_labels
    
    def load_all_data(self, segment_by_beat=True):
        """
        Load all ECG files and corresponding labels
        
        Args:
            segment_by_beat: If True, segment each recording into heartbeats
        
        Returns:
            X, y_encoded, patient_ids
        """
        print("\n" + "=" * 80)
        print("Loading ECG Data")
        print("=" * 80)
        
        # Load labels
        df_labels = self.load_labels()
        
        # Create 4-class mapping
        df_labels = self.create_4_class_mapping(df_labels)
        
        print(f"\n4-Class Distribution:")
        print(df_labels['Location_Class'].value_counts())
        
        # Get list of ECG files
        ecg_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        print(f"\n✓ Found {len(ecg_files)} ECG files")
        
        if segment_by_beat:
            print(f"✓ Segmentation mode: ENABLED (extracting individual heartbeats)")
            print(f"  Expected: ~10-20 heartbeats per file → ~{len(ecg_files)*10}-{len(ecg_files)*20} total samples")
        else:
            print(f"✓ Segmentation mode: DISABLED (using whole recordings)")
            print(f"  Expected: {len(ecg_files)} total samples")
        
        # Load ECG data
        X_data = []
        y_data = []
        patient_ids = []
        
        visualize_first = True  # Visualize first file for demonstration
        files_processed = 0
        total_segments = 0
        
        for i, filename in enumerate(ecg_files):
            if (i + 1) % 50 == 0:
                print(f"Processing {i + 1}/{len(ecg_files)} files... ({total_segments} segments so far)")
            
            # Extract patient ID from filename
            patient_id = int(filename.replace('.csv', ''))
            
            # Check if patient has label - use HospitalID column
            if 'HospitalID' in df_labels.columns:
                patient_label = df_labels[df_labels['HospitalID'] == patient_id]
            else:
                # Fallback to first column
                patient_label = df_labels[df_labels.iloc[:, 0].astype(str) == str(patient_id)]
            
            if len(patient_label) > 0 and 'Location_Class' in patient_label.columns:
                # Check if this patient has a valid location class
                if pd.notna(patient_label['Location_Class'].values[0]):
                    # Load ECG data (returns list of segments)
                    filepath = os.path.join(self.data_dir, filename)
                    ecg_segments = self.load_ecg_file(
                        filepath, 
                        segment_by_beat=segment_by_beat,
                        visualize_first=visualize_first
                    )
                    
                    visualize_first = False  # Only visualize first file
                    
                    if len(ecg_segments) > 0:
                        # Get label for this patient
                        label = patient_label['Location_Class'].values[0]
                        
                        # Add all segments with same label
                        for segment in ecg_segments:
                            X_data.append(segment)
                            y_data.append(label)
                            patient_ids.append(patient_id)
                        
                        files_processed += 1
                        total_segments += len(ecg_segments)
        
        print(f"\n✓ Successfully processed {files_processed} ECG files")
        print(f"✓ Total segments extracted: {total_segments}")
        
        if segment_by_beat:
            avg_segments = total_segments / files_processed if files_processed > 0 else 0
            print(f"✓ Average heartbeats per file: {avg_segments:.1f}")
        
        # Convert to numpy arrays
        X = np.array(X_data)
        y = np.array(y_data)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\n✓ Final data shape: {X.shape}")
        print(f"✓ Labels shape: {y_encoded.shape}")
        print(f"✓ Number of classes: {len(self.label_encoder.classes_)}")
        print(f"✓ Classes: {self.label_encoder.classes_}")
        
        # Show class distribution in final dataset
        print(f"\n✓ Final class distribution:")
        unique, counts = np.unique(y_encoded, return_counts=True)
        for cls_idx, count in zip(unique, counts):
            cls_name = self.label_encoder.classes_[cls_idx]
            percentage = (count / len(y_encoded)) * 100
            print(f"  {cls_name}: {count} samples ({percentage:.1f}%)")
        
        return X, y_encoded, patient_ids


# ============================================================================
# 2. ECG GAN FOR DATA AUGMENTATION
# ============================================================================

class ECGGAN:
    """GAN for generating synthetic ECG signals to balance dataset"""
    
    def __init__(self, signal_shape, latent_dim=100):
        """
        Args:
            signal_shape: Shape of ECG signal (timesteps, channels)
            latent_dim: Dimension of latent space
        """
        self.signal_shape = signal_shape
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()
    
    def build_generator(self):
        """Build generator network"""
        model = models.Sequential(name='Generator')
        
        # Calculate intermediate dimensions
        timesteps, channels = self.signal_shape
        
        # Start with smaller dense layer
        model.add(layers.Dense(256, input_dim=self.latent_dim))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization())
        
        model.add(layers.Dense(512))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization())
        
        # Reshape to 3D for Conv1DTranspose
        initial_timesteps = timesteps // 8  # Start with reduced timesteps
        model.add(layers.Dense(initial_timesteps * 64))
        model.add(layers.Reshape((initial_timesteps, 64)))
        
        # Upsample using Conv1DTranspose
        model.add(layers.Conv1DTranspose(128, kernel_size=4, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Conv1DTranspose(64, kernel_size=4, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Conv1DTranspose(32, kernel_size=4, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        
        # Final layer to get correct number of channels
        model.add(layers.Conv1D(channels, kernel_size=7, padding='same', activation='tanh'))
        
        # Ensure correct output shape
        model.add(layers.Lambda(lambda x: tf.image.resize(
            tf.expand_dims(x, -1), 
            [timesteps, channels]
        )[:, :, :, 0]))
        
        return model
    
    def build_discriminator(self):
        """Build discriminator network"""
        model = models.Sequential(name='Discriminator')
        
        # Input: ECG signal
        model.add(layers.Flatten(input_shape=self.signal_shape))
        
        model.add(layers.Dense(1024))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Dense(512))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))
        
        # Output: real/fake probability
        model.add(layers.Dense(1, activation='sigmoid'))
        
        return model
    
    def build_gan(self):
        """Build combined GAN model"""
        self.discriminator.compile(
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Freeze discriminator when training generator
        self.discriminator.trainable = False
        
        # GAN model
        gan_input = layers.Input(shape=(self.latent_dim,))
        generated_signal = self.generator(gan_input)
        gan_output = self.discriminator(generated_signal)
        
        gan = models.Model(gan_input, gan_output, name='GAN')
        gan.compile(
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )
        
        return gan
    
    def train(self, X_train, epochs=100, batch_size=32):
        """Train GAN"""
        print("\n" + "=" * 80)
        print("Training ECG GAN")
        print("=" * 80)
        
        # Normalize to [-1, 1] for tanh activation
        X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
        X_train = X_train * 2 - 1
        
        batch_count = X_train.shape[0] // batch_size
        
        for epoch in range(epochs):
            for batch_idx in range(batch_count):
                # Train Discriminator
                # Real samples
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                real_signals = X_train[idx]
                
                # Fake samples
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                fake_signals = self.generator.predict(noise, verbose=0)
                
                # Labels
                real_labels = np.ones((batch_size, 1))
                fake_labels = np.zeros((batch_size, 1))
                
                # Train discriminator
                d_loss_real = self.discriminator.train_on_batch(real_signals, real_labels)
                d_loss_fake = self.discriminator.train_on_batch(fake_signals, fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # Train Generator
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                g_loss = self.gan.train_on_batch(noise, real_labels)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")
        
        print("✓ GAN training completed")
    
    def generate_samples(self, n_samples, class_label=None):
        """Generate synthetic ECG samples"""
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        generated_signals = self.generator.predict(noise, verbose=0)
        
        # Denormalize from [-1, 1]
        generated_signals = (generated_signals + 1) / 2
        
        return generated_signals

# ============================================================================
# 3. ATTENTION MECHANISM
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
        # Compute attention scores
        e = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        
        # Apply attention weights
        output = inputs * a
        
        return tf.reduce_sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# ============================================================================
# 4. CNN-BiLSTM + ATTENTION MODEL
# ============================================================================

class CNNBiLSTMAttention:
    """CNN-BiLSTM with Attention for ECG classification"""
    
    def __init__(self, input_shape, num_classes):
        """
        Args:
            input_shape: Shape of input ECG signal (timesteps, channels)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        """Build CNN-BiLSTM-Attention model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # CNN layers for feature extraction
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
        
        # BiLSTM layers for temporal dependencies
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
        
        # Attention mechanism
        x = AttentionLayer()(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='CNN_BiLSTM_Attention')
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile model"""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()

# ============================================================================
# 5. TRAINING PIPELINE
# ============================================================================

def balance_dataset_with_gan(X_train, y_train, target_samples_per_class=None):
    """Balance dataset using GAN-generated samples"""
    print("\n" + "=" * 80)
    print("Balancing Dataset with GAN")
    print("=" * 80)
    
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    print(f"Original class distribution:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} samples")
    
    if target_samples_per_class is None:
        target_samples_per_class = max(class_counts)
    
    X_balanced = []
    y_balanced = []
    
    for cls in unique_classes:
        # Get samples for this class
        cls_indices = np.where(y_train == cls)[0]
        cls_samples = X_train[cls_indices]
        
        X_balanced.append(cls_samples)
        y_balanced.extend([cls] * len(cls_samples))
        
        # Generate additional samples if needed
        n_needed = target_samples_per_class - len(cls_samples)
        
        if n_needed > 0:
            print(f"\nGenerating {n_needed} samples for class {cls}...")
            
            # Train GAN on this class
            gan = ECGGAN(signal_shape=X_train.shape[1:], latent_dim=100)
            gan.train(cls_samples, epochs=100, batch_size=min(32, len(cls_samples)))
            
            # Generate synthetic samples
            synthetic_samples = gan.generate_samples(n_needed)
            
            X_balanced.append(synthetic_samples)
            y_balanced.extend([cls] * n_needed)
    
    X_balanced = np.vstack(X_balanced)
    y_balanced = np.array(y_balanced)
    
    print(f"\n✓ Balanced dataset shape: {X_balanced.shape}")
    print(f"✓ Balanced class distribution:")
    unique_classes, class_counts = np.unique(y_balanced, return_counts=True)
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} samples")
    
    return X_balanced, y_balanced

def train_model(X_train, y_train, X_val, y_val, num_classes, epochs=100):
    """Train CNN-BiLSTM-Attention model"""
    print("\n" + "=" * 80)
    print("Training CNN-BiLSTM-Attention Model")
    print("=" * 80)
    
    # Build model
    model_builder = CNNBiLSTMAttention(
        input_shape=X_train.shape[1:],
        num_classes=num_classes
    )
    model_builder.compile_model(learning_rate=0.001)
    
    print("\nModel Architecture:")
    model_builder.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'best_ecg_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    history = model_builder.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    return model_builder.model, history

# ============================================================================
# 6. EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate model performance"""
    print("\n" + "=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Test Accuracy: {accuracy:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_
    ))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
    
    return accuracy, y_pred

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    print("✓ Training history saved as 'training_history.png'")

# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    print("\n" + "=" * 80)
    print("ECG 12-Lead Location Prediction Pipeline")
    print("=" * 80)
    
    # Configuration
    DATA_DIR = r'D:\GAN\PVCVTECGData'
    LABEL_FILE = r'D:\GAN\Diagnosis.xlsx'
    TARGET_LENGTH = 300  # 300 samples = 600ms at 500Hz (1 heartbeat)
    EPOCHS = 100
    USE_GAN_BALANCING = True
    USE_SEGMENTATION = True  # Enable heartbeat segmentation
    
    print(f"\n{'='*80}")
    print(f"Configuration:")
    print(f"{'='*80}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Label File: {LABEL_FILE}")
    print(f"Target Length: {TARGET_LENGTH} samples ({'600ms per heartbeat' if USE_SEGMENTATION else '10s whole recording'})")
    print(f"Epochs: {EPOCHS}")
    print(f"GAN Balancing: {USE_GAN_BALANCING}")
    print(f"Heartbeat Segmentation: {USE_SEGMENTATION}")
    
    # 1. Load Data
    loader = ECGDataLoader(DATA_DIR, LABEL_FILE, target_length=TARGET_LENGTH)
    X, y, patient_ids = loader.load_all_data(segment_by_beat=USE_SEGMENTATION)

    
    # 2. Split Data
    print("\n" + "=" * 80)
    print("Splitting Dataset")
    print("=" * 80)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Validation set: {X_val.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    # 3. Balance Dataset with GAN (optional)
    if USE_GAN_BALANCING:
        X_train, y_train = balance_dataset_with_gan(X_train, y_train)
        
        # Save balanced training data for use by other models
        print("\n" + "=" * 80)
        print("Saving GAN-Balanced Dataset")
        print("=" * 80)
        
        # Combine all data (train + val + test) with balanced training set
        X_all_balanced = np.vstack([X_train, X_val, X_test])
        y_all_balanced = np.concatenate([y_train, y_val, y_test])
        
        np.save('X_preprocessed.npy', X_all_balanced)
        np.save('y_preprocessed.npy', y_all_balanced)
        
        print(f"✓ Saved X_preprocessed.npy: {X_all_balanced.shape}")
        print(f"✓ Saved y_preprocessed.npy: {y_all_balanced.shape}")
        print("✓ Dataset sudah di-balance dengan GAN dan siap digunakan oleh model lain")
    
    # 4. Train Model
    num_classes = len(np.unique(y))
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        num_classes=num_classes,
        epochs=EPOCHS
    )
    
    # 5. Plot Training History
    plot_training_history(history)
    
    # 6. Evaluate Model
    accuracy, y_pred = evaluate_model(model, X_test, y_test, loader.label_encoder)
    
    # 7. Save Model and Artifacts
    print("\n" + "=" * 80)
    print("Saving Artifacts")
    print("=" * 80)
    
    model.save('ecg_location_model_final.h5')
    print("✓ Model saved as 'ecg_location_model_final.h5'")
    
    # Save label encoder
    import joblib
    joblib.dump(loader.label_encoder, 'label_encoder_location.joblib')
    print("✓ Label encoder saved as 'label_encoder_location.joblib'")
    
    # Save predictions
    results_df = pd.DataFrame({
        'True_Label': loader.label_encoder.inverse_transform(y_test),
        'Predicted_Label': loader.label_encoder.inverse_transform(y_pred)
    })
    results_df.to_csv('predictions.csv', index=False)
    print("✓ Predictions saved as 'predictions.csv'")
    
    print("\n" + "=" * 80)
    print("Pipeline Completed Successfully!")
    print("=" * 80)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print(f"\n📁 File X_preprocessed.npy dan y_preprocessed.npy sudah tersimpan")
    print(f"📁 Ketiga model (CNN-BiLSTM, CNN-DNN, CNN-RNN) dapat menggunakan data yang sudah di-balance dengan GAN")

if __name__ == "__main__":
    main()
