"""
Fix BiLSTM GridSearch Visualization
Regenerate PNG from saved CSV files
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("="*80)
print("REGENERATING BiLSTM GRIDSEARCH VISUALIZATION")
print("="*80)

# Load saved data
print("\nLoading saved results...")
df_screening = pd.read_csv('GridSearch_Stage1_Screening_BiLSTM.csv')
df_folds = pd.read_csv('GridSearch_Stage2_FoldResults_BiLSTM.csv')

print(f"✓ Stage 1: {len(df_screening)} configurations")
print(f"✓ Stage 2: {len(df_folds)} folds")

# Get best accuracy from screening
best_acc_stage1 = df_screening['accuracy'].max()
mean_acc_stage2 = df_folds['accuracy'].mean()

print(f"\nBest Stage 1 accuracy: {best_acc_stage1:.4f}")
print(f"Mean Stage 2 accuracy: {mean_acc_stage2:.4f}")

# Create visualization
plt.figure(figsize=(14, 10))

# Plot 1: Screening results
plt.subplot(2, 2, 1)
x_pos = np.arange(len(df_screening))
plt.bar(x_pos, df_screening['accuracy'], alpha=0.7, color='steelblue')
plt.axhline(y=best_acc_stage1, color='r', linestyle='--', 
            label=f'Best: {best_acc_stage1:.4f}')
plt.xlabel('Configuration Index', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Stage 1: All Configurations Screening - BiLSTM', 
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Plot 2: Fold results
plt.subplot(2, 2, 2)
plt.bar(df_folds['fold'], df_folds['accuracy'], alpha=0.7, color='forestgreen')
plt.axhline(y=mean_acc_stage2, color='r', linestyle='--', 
            label=f'Mean: {mean_acc_stage2:.4f}')
plt.xlabel('Fold', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Stage 2: Best Config 5-Fold CV - BiLSTM', 
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Plot 3: Learning rate impact
plt.subplot(2, 2, 3)
lr_groups = df_screening.groupby('learning_rate')['accuracy'].mean()
plt.bar(range(len(lr_groups)), lr_groups.values, alpha=0.7, color='darkorange')
plt.xlabel('Learning Rate', fontsize=12)
plt.ylabel('Mean Accuracy', fontsize=12)
plt.title('Learning Rate Impact - BiLSTM', fontsize=14, fontweight='bold')
plt.xticks(range(len(lr_groups)), [f'{lr:.5f}' for lr in lr_groups.index])
plt.grid(True, alpha=0.3, axis='y')

# Plot 4: Dropout impact
plt.subplot(2, 2, 4)
dropout_groups = df_screening.groupby('dropout_rate')['accuracy'].mean()
plt.bar(range(len(dropout_groups)), dropout_groups.values, alpha=0.7, color='crimson')
plt.xlabel('Dropout Rate', fontsize=12)
plt.ylabel('Mean Accuracy', fontsize=12)
plt.title('Dropout Impact - BiLSTM', fontsize=14, fontweight='bold')
plt.xticks(range(len(dropout_groups)), [f'{d:.1f}' for d in dropout_groups.index])
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('GridSearch_Complete_Visualization_BiLSTM.png', dpi=300)
print(f"\n✓ Visualization saved: GridSearch_Complete_Visualization_BiLSTM.png")

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nStage 1 (Quick Screening):")
print(f"  Best accuracy: {best_acc_stage1:.4f}")
print(f"  Worst accuracy: {df_screening['accuracy'].min():.4f}")
print(f"  Range: {(best_acc_stage1 - df_screening['accuracy'].min()):.4f}")

print(f"\nStage 2 (Full Validation):")
print(f"  Mean accuracy: {mean_acc_stage2:.4f} ± {df_folds['accuracy'].std():.4f}")
print(f"  Min accuracy: {df_folds['accuracy'].min():.4f}")
print(f"  Max accuracy: {df_folds['accuracy'].max():.4f}")

print(f"\nBest Parameters:")
best_row = df_screening.iloc[0]  # Already sorted by accuracy
print(f"  learning_rate: {best_row['learning_rate']}")
print(f"  dropout_rate: {best_row['dropout_rate']}")
print(f"  batch_size: {int(best_row['batch_size'])}")
print(f"  lstm_units: {int(best_row['lstm_units'])}")

print("\n" + "="*80)
print("✅ DONE!")
print("="*80)
