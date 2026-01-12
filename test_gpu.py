import tensorflow as tf
import sys

print("="*80)
print("🔍 GPU DETECTION TEST")
print("="*80)

print(f"\nPython Version: {sys.version}")
print(f"TensorFlow Version: {tf.__version__}")

# Check GPU
print("\n" + "="*80)
print("GPU DEVICES")
print("="*80)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n✅ GPU DETECTED: {len(gpus)} device(s)\n")
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}:")
        print(f"  Name: {gpu.name}")
        print(f"  Type: {gpu.device_type}")
        
        # Get GPU details
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            print(f"  Details: {gpu_details}")
        except:
            pass
        print()
    
    # Test GPU computation
    print("="*80)
    print("GPU COMPUTATION TEST")
    print("="*80)
    
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print("\n✅ GPU computation successful!")
            print(f"Result:\n{c.numpy()}")
    except Exception as e:
        print(f"\n❌ GPU computation failed: {e}")
    
    # Enable mixed precision
    print("\n" + "="*80)
    print("MIXED PRECISION TEST")
    print("="*80)
    
    try:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f"\n✅ Mixed Precision enabled: {policy.name}")
        print(f"   Compute dtype: {policy.compute_dtype}")
        print(f"   Variable dtype: {policy.variable_dtype}")
        print("\n   This will make training 2-3x faster on GPU!")
    except Exception as e:
        print(f"\n⚠️  Mixed precision not available: {e}")
    
else:
    print("\n⚠️  NO GPU DETECTED")
    print("\nRunning on CPU. Training will be slower.")
    print("\nTo enable GPU:")
    print("1. Install CUDA Toolkit")
    print("2. Install cuDNN")
    print("3. Install tensorflow-gpu or tensorflow (with GPU support)")

# Check CUDA
print("\n" + "="*80)
print("CUDA AVAILABILITY")
print("="*80)

cuda_available = tf.test.is_built_with_cuda()
print(f"\nTensorFlow built with CUDA: {cuda_available}")

if cuda_available:
    print("✅ CUDA support is available")
else:
    print("⚠️  CUDA support not available")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if gpus:
    print("\n✅ YOUR SYSTEM IS READY FOR GPU TRAINING!")
    print("\nRecommended batch size: 64")
    print("Expected speedup: 2-3x faster than CPU")
    print("\nYou can run: python gan_fixed.py")
else:
    print("\n⚠️  GPU NOT DETECTED")
    print("\nOptions:")
    print("1. Install GPU drivers and CUDA")
    print("2. Use Google Colab (free GPU)")
    print("3. Use Kaggle Notebooks (free GPU)")
    print("\nYou can still run on CPU, but it will be slower.")
    print("Recommended batch size for CPU: 32")

print("\n" + "="*80)
