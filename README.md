# ECG Location Prediction with Deep Learning

Prediksi lokasi PVC (Premature Ventricular Contraction) menggunakan Deep Learning dengan GAN-based data augmentation dan hyperparameter tuning.

## 🎯 Project Overview

Project ini menggunakan data ECG untuk memprediksi lokasi asal PVC dengan 4 kategori:
- Left Ventricular Region
- Outflow Tract Region  
- Right Ventricular Region
- Septal Region

## 🏆 Results

### Model Performance (After GridSearch)

| Model | Accuracy | F1-Score | Stability |
|-------|----------|----------|-----------|
| **CNN-DNN** 🥇 | **91.25%** | **91.67%** | ±0.26% |
| **CNN-BiLSTM** 🥈 | 90.98% | 91.37% | ±0.70% |
| **CNN-RNN** 🥉 | 87.29% | 87.92% | ±2.32% |

### Key Improvements
- **Heartbeat Segmentation**: 334 → 1,066 samples (+219%)
- **GAN Balancing**: 1,066 → 12,356 samples (+1,059%)
- **GridSearch Optimization**: +1-5% accuracy improvement
- **Data Balance**: Imbalance ratio 13:1 → 1.39:1

## 📁 Project Structure

```
D:\GAN\
├── ecg_location_prediction_complete.py  # Main preprocessing & GAN
├── model_cnn_bilstm_attention.py        # BiLSTM default model
├── model_cnn_dnn.py                     # DNN default model
├── model_cnn_rnn_attention.py           # RNN default model
├── model_cnn_bilstm_gridsearch.py       # BiLSTM GridSearch
├── model_cnn_dnn_gridsearch.py          # DNN GridSearch
├── model_cnn_rnn_gridsearch.py          # RNN GridSearch
├── GridSearch_*.csv                     # GridSearch results
├── GridSearch_*.png                     # Visualizations
└── GridSearch_*.txt                     # Summaries
```

## 🚀 Quick Start

### 1. Preprocessing & GAN Balancing
```bash
python ecg_location_prediction_complete.py
```
Output:
- `X_preprocessed.npy` (12,356 samples)
- `y_preprocessed.npy`
- `label_encoder_location.joblib`

### 2. Train Default Models
```bash
# Run in parallel (3 terminals)
python model_cnn_bilstm_attention.py
python model_cnn_dnn.py
python model_cnn_rnn_attention.py
```

### 3. GridSearch Hyperparameter Tuning
```bash
# Run in parallel (3 terminals)
python model_cnn_bilstm_gridsearch.py
python model_cnn_dnn_gridsearch.py
python model_cnn_rnn_gridsearch.py
```

## 📊 Methodology

### 1. Data Preprocessing
- **Heartbeat Segmentation**: R-peak detection → 600ms windows
- **Normalization**: Min-max scaling
- **Augmentation**: GAN-based synthetic data generation

### 2. Model Architectures

#### CNN-DNN (Best Model)
- 3 CNN blocks (64, 128, 256 filters)
- 4 Dense layers (512, 256, 128, 64 units)
- Batch Normalization + Dropout

#### CNN-BiLSTM + Attention
- 3 CNN blocks
- 2 Bidirectional LSTM layers
- Attention mechanism
- 2 Dense layers

#### CNN-RNN + Attention
- 3 CNN blocks
- 2 SimpleRNN layers
- Attention mechanism
- 2 Dense layers

### 3. GridSearch Strategy

**2-Stage Approach:**
- **Stage 1**: Quick screening (20 epochs, 1 split) → 16 configs
- **Stage 2**: Full validation (50 epochs, 5-fold CV) → Best config

**Parameter Grid:**
```python
{
    'learning_rate': [0.001, 0.0001],
    'dropout_rate': [0.3, 0.5],
    'batch_size': [32, 64],
    'units': [128, 64] or [512, 256]
}
```

## 📈 Key Findings

### 1. Heartbeat Segmentation Impact
- Increased samples from 334 → 1,066 (+219%)
- Improved accuracy from 64% → 82% (+17.4%)
- All classes now predictable (was 2 classes at 0%)

### 2. GAN Balancing Impact
- Increased samples to 12,356 (+1,059%)
- Balanced distribution: 21-30% per class
- Imbalance ratio: 13:1 → 1.39:1

### 3. GridSearch Optimization
- **DNN**: +4-5% improvement (MASSIVE!)
- **BiLSTM**: +1% improvement
- **RNN**: +1-2% improvement

### 4. Best Configurations Found

**CNN-DNN (Winner):**
- batch_size: 64 (vs default 32)
- dense_units: 256 (vs default 512)
- Simpler model, better generalization!

**CNN-BiLSTM:**
- batch_size: 64 (vs default 32)
- lstm_units: 64 (vs default 128)
- Smaller model, less overfitting!

## 🛠️ Requirements

```
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
joblib>=1.0.0
```

## 📝 Citation

If you use this code, please cite:
```
@misc{ecg_pvc_prediction_2026,
  author = {Your Name},
  title = {ECG Location Prediction with Deep Learning and GAN Augmentation},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/znashif/pvc}
}
```

## 📧 Contact

For questions or collaborations, please open an issue or contact [your email].

## 📄 License

This project is licensed under the MIT License.

---

**Note**: Data files (`.npy`, `.h5`, `.xlsx`) are excluded from this repository due to size constraints. Please contact for access to the dataset.
