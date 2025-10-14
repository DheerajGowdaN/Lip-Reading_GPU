""""""

GPU Verification Script for Multi-Lingual Lip Reading SystemGPU Verification Script for Multi-Lingual Lip Reading System

Checks TensorFlow GPU configuration and dependenciesChecks TensorFlow GPU configuration and dependencies

Author: AI AssistantAuthor: AI Assistant

Date: October 2025Date: October 2025

""""""



import sysimport sys

import platformimport platform



print("="*70)print("="*70)

print("GPU VERIFICATION REPORT")print("GPU VERIFICATION REPORT")

print("="*70)print("="*70)

print()print()



# System Info# System Info

print("📋 SYSTEM INFORMATION:")print("📋 SYSTEM INFORMATION:")

print(f"  Python Version: {sys.version}")print(f"  Python Version: {sys.version}")

print(f"  Platform: {platform.platform()}")print(f"  Platform: {platform.platform()}")

print(f"  Architecture: {platform.machine()}")print(f"  Architecture: {platform.machine()}")

print()print()



# Check TensorFlow# Check TensorFlow

print("🔍 CHECKING TENSORFLOW:")print("🔍 CHECKING TENSORFLOW:")

try:try:

    import tensorflow as tf    import tensorflow as tf

    print(f"  ✓ TensorFlow installed: {tf.__version__}")    print(f"  ✓ TensorFlow installed: {tf.__version__}")

        

    # Check if TensorFlow was built with CUDA    # Check if TensorFlow was built with CUDA

    cuda_built = tf.test.is_built_with_cuda()    cuda_built = tf.test.is_built_with_cuda()

    print(f"  Built with CUDA: {cuda_built}")    print(f"  Built with CUDA: {cuda_built}")

        

    if not cuda_built:    if not cuda_built:

        print("  ⚠️  WARNING: TensorFlow was NOT built with CUDA support!")        print("  ⚠️  WARNING: TensorFlow was NOT built with CUDA support!")

        print("     Your current TensorFlow installation cannot use GPU.")        print("     Your current TensorFlow installation cannot use GPU.")

        print()        print()

        print("  📦 SOLUTION:")        print("  📦 SOLUTION:")

        print("     Recreate conda environment with GPU support")        print("     Install TensorFlow with GPU support:")

except ImportError as e:        print("     pip uninstall tensorflow")

    print(f"  ✗ TensorFlow not installed: {e}")        print("     pip install tensorflow[and-cuda]")

    sys.exit(1)        print("     OR")

        print("     pip install tensorflow-gpu")

print()except ImportError as e:

    print(f"  ✗ TensorFlow not installed: {e}")

# Check GPU Devices    sys.exit(1)

print("🖥️  GPU DEVICES:")

try:print()

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:# Check GPU Devices

        print(f"  ✓ Found {len(gpus)} GPU(s):")print("🖥️  GPU DEVICES:")

        for i, gpu in enumerate(gpus):try:

            print(f"    [{i}] {gpu.name}")    gpus = tf.config.list_physical_devices('GPU')

            print(f"        Type: {gpu.device_type}")    if gpus:

                print(f"  ✓ Found {len(gpus)} GPU(s):")

        # Try to get GPU details        for i, gpu in enumerate(gpus):

        try:            print(f"    [{i}] {gpu.name}")

            from tensorflow.python.client import device_lib            print(f"        Type: {gpu.device_type}")

            local_devices = device_lib.list_local_devices()        

            for device in local_devices:        # Try to get GPU details

                if device.device_type == 'GPU':        try:

                    print(f"    Details:")            from tensorflow.python.client import device_lib

                    print(f"      Memory: {device.memory_limit / (1024**3):.2f} GB")            local_devices = device_lib.list_local_devices()

                    print(f"      Compute Capability: {device.physical_device_desc}")            for device in local_devices:

        except Exception as e:                if device.device_type == 'GPU':

            print(f"    (Could not get detailed GPU info: {e})")                    print(f"    Details:")

    else:                    print(f"      Memory: {device.memory_limit / (1024**3):.2f} GB")

        print("  ✗ No GPU devices found!")                    print(f"      Compute Capability: {device.physical_device_desc}")

        print()        except Exception as e:

        print("  🔧 TROUBLESHOOTING:")            print(f"    (Could not get detailed GPU info: {e})")

        print("     1. Check if GPU is CUDA-compatible (NVIDIA only)")    else:

        print("     2. Install NVIDIA GPU drivers")        print("  ✗ No GPU devices found!")

        print("     3. Recreate conda environment with GPU support")        print()

except Exception as e:        print("  🔧 TROUBLESHOOTING:")

    print(f"  ✗ Error checking GPU: {e}")        print("     1. Check if GPU is CUDA-compatible (NVIDIA only)")

        print("     2. Install NVIDIA GPU drivers")

print()        print("     3. Install CUDA Toolkit (11.8 or 12.x)")

        print("     4. Install cuDNN library")

# Check CUDA availability        print("     5. Install TensorFlow with GPU support:")

print("🔌 CUDA AVAILABILITY:")        print("        pip install tensorflow[and-cuda]")

try:except Exception as e:

    cuda_available = tf.test.is_gpu_available(cuda_only=True)    print(f"  ✗ Error checking GPU: {e}")

    print(f"  CUDA GPU Available: {cuda_available}")

except:print()

    # Try newer method

    try:# Check CUDA availability

        gpus = tf.config.list_physical_devices('GPU')print("🔌 CUDA AVAILABILITY:")

        cuda_available = len(gpus) > 0try:

        print(f"  CUDA GPU Available: {cuda_available}")    cuda_available = tf.test.is_gpu_available(cuda_only=True)

    except:    print(f"  CUDA GPU Available: {cuda_available}")

        print("  Could not determine CUDA availability")except:

    # Try newer method

print()    try:

        gpus = tf.config.list_physical_devices('GPU')

# Test GPU computation        cuda_available = len(gpus) > 0

print("🧪 TESTING GPU COMPUTATION:")        print(f"  CUDA GPU Available: {cuda_available}")

try:    except:

    gpus = tf.config.list_physical_devices('GPU')        print("  Could not determine CUDA availability")

    if gpus:

        # Create a simple operation on GPUprint()

        with tf.device('/GPU:0'):

            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])# Test GPU computation

            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])print("🧪 TESTING GPU COMPUTATION:")

            c = tf.matmul(a, b)try:

            print(f"  ✓ GPU computation successful!")    gpus = tf.config.list_physical_devices('GPU')

            print(f"    Test result: {c.numpy()}")    if gpus:

                    # Create a simple operation on GPU

        # Check device placement        with tf.device('/GPU:0'):

        print(f"  ✓ TensorFlow can place operations on GPU")            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])

    else:            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

        print("  ✗ No GPU available for testing")            c = tf.matmul(a, b)

except Exception as e:            print(f"  ✓ GPU computation successful!")

    print(f"  ✗ GPU computation failed: {e}")            print(f"    Test result: {c.numpy()}")

            

print()        # Check device placement

        print(f"  ✓ TensorFlow can place operations on GPU")

# Check Mixed Precision    else:

print("🎯 MIXED PRECISION:")        print("  ✗ No GPU available for testing")

policy = Noneexcept Exception as e:

try:    print(f"  ✗ GPU computation failed: {e}")

    policy = tf.keras.mixed_precision.global_policy()

    print(f"  Current Policy: {policy.name}")print()

    

    if policy.name == 'mixed_float16':# Check Mixed Precision

        print(f"  ✓ Mixed precision (FP16) is enabled")print("🎯 MIXED PRECISION:")

    else:policy = None

        print(f"  ℹ️  Mixed precision not enabled (using {policy.name})")try:

        print(f"     This is normal before training starts")    policy = tf.keras.mixed_precision.global_policy()

except Exception as e:    print(f"  Current Policy: {policy.name}")

    print(f"  Could not check mixed precision: {e}")    

    if policy.name == 'mixed_float16':

print()        print(f"  ✓ Mixed precision (FP16) is enabled")

    else:

# Check Memory Growth        print(f"  ℹ️  Mixed precision not enabled (using {policy.name})")

print("💾 GPU MEMORY CONFIGURATION:")        print(f"     This is normal before training starts")

try:except Exception as e:

    gpus = tf.config.list_physical_devices('GPU')    print(f"  Could not check mixed precision: {e}")

    if gpus:

        for gpu in gpus:print()

            try:

                memory_growth = tf.config.experimental.get_memory_growth(gpu)# Check Memory Growth

                print(f"  {gpu.name}")print("💾 GPU MEMORY CONFIGURATION:")

                print(f"    Memory Growth: {memory_growth}")try:

            except:    gpus = tf.config.list_physical_devices('GPU')

                print(f"  {gpu.name}")    if gpus:

                print(f"    Memory Growth: (cannot determine)")        for gpu in gpus:

    else:            try:

        print("  No GPU devices configured")                memory_growth = tf.config.experimental.get_memory_growth(gpu)

except Exception as e:                print(f"  {gpu.name}")

    print(f"  Error checking memory: {e}")                print(f"    Memory Growth: {memory_growth}")

            except:

print()                print(f"  {gpu.name}")

                print(f"    Memory Growth: (cannot determine)")

# Check other required packages    else:

print("📦 REQUIRED PACKAGES:")        print("  No GPU devices configured")

packages = [except Exception as e:

    ('numpy', 'numpy'),    print(f"  Error checking memory: {e}")

    ('opencv-python', 'cv2'),

    ('mediapipe', 'mediapipe'),print()

    ('matplotlib', 'matplotlib'),

    ('sklearn', 'sklearn'),# Check other required packages

]print("📦 REQUIRED PACKAGES:")

packages = [

for name, import_name in packages:    ('numpy', 'numpy'),

    try:    ('opencv-python', 'cv2'),

        module = __import__(import_name)    ('mediapipe', 'mediapipe'),

        version = getattr(module, '__version__', 'unknown')    ('matplotlib', 'matplotlib'),

        print(f"  ✓ {name}: {version}")    ('sklearn', 'sklearn'),

    except ImportError:    ('PIL', 'PIL'),

        print(f"  ✗ {name}: NOT INSTALLED")]



print()for name, import_name in packages:

    try:

# Recommendations        module = __import__(import_name)

print("="*70)        version = getattr(module, '__version__', 'unknown')

print("📝 RECOMMENDATIONS:")        print(f"  ✓ {name}: {version}")

print("="*70)    except ImportError:

        print(f"  ✗ {name}: NOT INSTALLED")

gpus = tf.config.list_physical_devices('GPU')

cuda_built = tf.test.is_built_with_cuda()print()



if not cuda_built:# Recommendations

    print()print("="*70)

    print("⚠️  CRITICAL: TensorFlow not built with CUDA")print("📝 RECOMMENDATIONS:")

    print()print("="*70)

    print("   Your TensorFlow installation cannot use GPU.")

    print("   Recreate conda environment:")gpus = tf.config.list_physical_devices('GPU')

    print()cuda_built = tf.test.is_built_with_cuda()

    print("   conda deactivate")

    print("   conda env remove -n lipread_gpu")if not cuda_built:

    print("   conda create -n lipread_gpu python=3.9 tensorflow-gpu=2.6.0 \\")    print()

    print("       cudatoolkit=11.3.1 cudnn=8.2.1 -c conda-forge -y")    print("⚠️  CRITICAL: TensorFlow not built with CUDA")

    print()    print()

elif not gpus:    print("   Your TensorFlow installation cannot use GPU.")

    print()    print("   You need to reinstall TensorFlow with GPU support.")

    print("⚠️  No GPU devices detected")    print()

    print()    

    print("   Possible causes:")    # Platform-specific recommendations

    print("   1. No NVIDIA GPU in system")    is_windows = platform.system() == 'Windows'

    print("   2. GPU drivers not installed")    

    print("   3. CUDA toolkit not installed")    if is_windows:

    print()        print("   🪟 WINDOWS SOLUTION (Choose ONE):")

    print("   STEPS FOR NVIDIA GPU:")        print()

    print("   1. Install latest NVIDIA GPU drivers:")        print("   Option 1: Upgrade to TensorFlow 2.15+ (RECOMMENDED - EASIEST)")

    print("      https://www.nvidia.com/Download/index.aspx")        print("   ---------------------------------------------------------")

    print()        print("   TensorFlow 2.15+ includes CUDA libraries on Windows!")

    print("   2. Recreate conda environment with GPU support")        print()

    print()        print("   pip uninstall -y tensorflow tensorflow-intel")

else:        print("   pip install tensorflow==2.15.0")

    print()        print()

    print("✅ GPU configuration looks good!")        print("   Option 2: Manual CUDA Installation (Complex)")

    print()        print("   --------------------------------------------")

    print("   Your system has GPU(s) and TensorFlow can use them.")        print("   1. Install CUDA Toolkit 11.8")

    print("   Make sure to select 'GPU' in the training GUI dropdown.")        print("   2. Install cuDNN 8.6")

    print()        print("   3. Keep TensorFlow 2.13")

    if policy and policy.name != 'mixed_float16':        print("   See WINDOWS_GPU_SETUP.md for detailed steps")

        print("   ℹ️  Mixed precision will be enabled when training starts.")        print()

    print()        print("   ⚠️  Note: tensorflow[and-cuda] does NOT work on Windows!")

    else:

print("="*70)        print("   🐧 LINUX/MAC SOLUTION:")

print()        print("   1. Uninstall current TensorFlow:")

        print("      pip uninstall tensorflow tensorflow-intel")

# Test with model compilation        print()

print("🔬 TESTING MODEL COMPILATION:")        print("   2. Install TensorFlow with GPU support:")

try:        print("      pip install tensorflow[and-cuda]")

    from src.model import LipReadingModel        print()

            print("   3. Restart Python/Application")

    print("  Creating test model...")    print()

    model = LipReadingModel(num_classes=10, sequence_length=75, num_features=100)elif not gpus:

    model.build_model(num_features=100)    print()

        print("⚠️  No GPU devices detected")

    print("  Compiling with GPU...")    print()

    model.compile_model(learning_rate=0.001, device='GPU')    print("   Possible causes:")

        print("   1. No NVIDIA GPU in system")

    print()    print("   2. GPU drivers not installed")

    print("  ✓ Model compilation test successful!")    print("   3. CUDA toolkit not installed")

    print("    If you saw GPU messages above, your setup is correct.")    print("   4. cuDNN library not installed")

        print()

except Exception as e:    print("   STEPS FOR NVIDIA GPU:")

    print(f"  ✗ Model compilation test failed: {e}")    print("   1. Install latest NVIDIA GPU drivers:")

    print()    print("      https://www.nvidia.com/Download/index.aspx")

    print("  This might indicate an issue with the model code or dependencies.")    print()

    print("   2. Install CUDA Toolkit (11.8 or 12.x):")

print()    print("      https://developer.nvidia.com/cuda-downloads")

print("="*70)    print()

print("VERIFICATION COMPLETE")    print("   3. TensorFlow 2.13+ includes cuDNN, but verify:")

print("="*70)    print("      https://developer.nvidia.com/cudnn")

print()    print()

print("💡 TIP: Run this script before training to verify GPU setup:")    print("   4. Reinstall TensorFlow:")

print("   python verify_gpu.py")    print("      pip install --upgrade tensorflow[and-cuda]")

print()    print()

else:
    print()
    print("✅ GPU configuration looks good!")
    print()
    print("   Your system has GPU(s) and TensorFlow can use them.")
    print("   Make sure to select 'GPU' in the training GUI dropdown.")
    print()
    if policy and policy.name != 'mixed_float16':
        print("   ℹ️  Mixed precision will be enabled when training starts.")
    print()

print("="*70)
print()

# Test with model compilation
print("🔬 TESTING MODEL COMPILATION:")
try:
    from src.model import LipReadingModel
    
    print("  Creating test model...")
    model = LipReadingModel(num_classes=10, sequence_length=75, num_features=100)
    model.build_model(num_features=100)
    
    print("  Compiling with GPU...")
    model.compile_model(learning_rate=0.001, device='GPU')
    
    print()
    print("  ✓ Model compilation test successful!")
    print("    If you saw GPU messages above, your setup is correct.")
    
except Exception as e:
    print(f"  ✗ Model compilation test failed: {e}")
    print()
    print("  This might indicate an issue with the model code or dependencies.")

print()
print("="*70)
print("VERIFICATION COMPLETE")
print("="*70)
print()
print("💡 TIP: Run this script before training to verify GPU setup:")
print("   python verify_gpu.py")
print()
