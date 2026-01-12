"""
Generate Fixed Confusion Matrix for RNN with Custom Attention Layer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import tensorflow as tf
from tensorflow import keras

# Define Attention Layer (needed for loading model)
class AttentionLayer(keras.layers.Layer):
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

print("="*70)
print("GENERATING FIXED CONFUSION MATRIX FOR CNN-RNN")
print("="*70)

# Load data
print("\nLoading data...")
X = np.load('X_preprocessed.npy')
y = np.load('y_preprocessed.npy')
le = joblib.load('label_encoder_location.joblib')

print(f"Total samples: {len(y)}")

# Use last 20% as test set
test_size = int(0.2 * len(X))
X_test = X[-test_size:]
y_test = y[-test_size:]

print(f"Test set size: {len(y_test)}")

# Load RNN model with custom objects
print("\nLoading RNN model...")

rnn_paths = [
    'best_model_cnnrnn_default_gan.h5',
    'rnn default/best_model_cnnrnn_default_gan.h5'
]

model_rnn = None
for path in rnn_paths:
    try:
        model_rnn = keras.models.load_model(
            path,
            custom_objects={'AttentionLayer': AttentionLayer},
            compile=False
        )
        print(f"Loaded model from: {path}")
        break
    except Exception as e:
        print(f"Failed to load from {path}: {e}")
        continue

if model_rnn is None:
    print("ERROR: Could not load RNN model")
    exit(1)

# Make predictions
print("\nMaking predictions...")
y_pred = np.argmax(model_rnn.predict(X_test, verbose=0), axis=1)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# Create confusion matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))

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
    vmin=0
)

plt.title(f'Confusion Matrix - CNN-RNN + Attention\\nAccuracy: {acc:.4f}',
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()

filename = 'CM_CNNRNN_FIXED.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\nSaved: {filename}")
print(f"Accuracy: {acc:.4f}")

print("\n" + "="*70)
print("DONE!")
print("="*70)
