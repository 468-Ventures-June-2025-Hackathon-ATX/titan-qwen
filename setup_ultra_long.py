#!/usr/bin/env python3
"""
Setup script for Ultra-Long Context TITANS + Qwen implementation
Installs dependencies and sets up environment for 512K token context windows
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"🔧 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {description}")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python 3.11+ is available"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} found, but 3.11+ required")
        return False

def check_system_requirements():
    """Check system requirements for ultra-long context"""
    print("💻 Checking system requirements...")
    
    # Check OS
    system = platform.system()
    print(f"OS: {system}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        print(f"Total Memory: {total_gb:.1f}GB")
        print(f"Available Memory: {available_gb:.1f}GB")
        
        if total_gb >= 32:
            print("✅ Sufficient memory for ultra-long contexts")
            if total_gb >= 64:
                print("🚀 Excellent! 64GB+ detected - optimal for 512K contexts")
            return True
        else:
            print("⚠️  Warning: Less than 32GB RAM detected. Ultra-long contexts may be limited.")
            return False
            
    except ImportError:
        print("⚠️  Could not check memory (psutil not installed)")
        return True

def setup_virtual_environment():
    """Set up Python virtual environment"""
    print("📦 Setting up virtual environment...")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    # Create virtual environment with Python 3.11
    success = run_command(
        "python3.11 -m venv venv",
        "Creating virtual environment with Python 3.11"
    )
    
    if not success:
        # Fallback to default python
        success = run_command(
            "python -m venv venv",
            "Creating virtual environment with default Python"
        )
    
    return success

def install_dependencies():
    """Install required dependencies"""
    print("📚 Installing dependencies...")
    
    # Determine pip command based on OS
    if platform.system() == "Windows":
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    # Upgrade pip first
    run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")
    
    # Core dependencies
    dependencies = [
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "titans-pytorch",
        "accelerate",
        "psutil",
        "numpy",
        "tokenizers",
        "safetensors",
        "huggingface-hub"
    ]
    
    # Install PyTorch with appropriate backend
    system = platform.system()
    if system == "Darwin":  # macOS
        print("🍎 Installing PyTorch for macOS with MPS support...")
        torch_install = run_command(
            f"{pip_cmd} install torch torchvision torchaudio",
            "Installing PyTorch for macOS"
        )
    else:
        print("🔥 Installing PyTorch with CUDA support...")
        torch_install = run_command(
            f"{pip_cmd} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "Installing PyTorch with CUDA"
        )
    
    if not torch_install:
        print("⚠️  PyTorch installation failed, trying CPU-only version...")
        run_command(
            f"{pip_cmd} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
            "Installing PyTorch CPU-only"
        )
    
    # Install other dependencies
    for dep in dependencies[1:]:  # Skip torch since we installed it separately
        run_command(f"{pip_cmd} install {dep}", f"Installing {dep}")
    
    # Install optional dependencies for better performance
    optional_deps = [
        "flash-attn",  # For flash attention (if supported)
        "bitsandbytes",  # For quantization
    ]
    
    for dep in optional_deps:
        print(f"🔧 Attempting to install optional dependency: {dep}")
        success = run_command(f"{pip_cmd} install {dep}", f"Installing {dep} (optional)")
        if not success:
            print(f"⚠️  Optional dependency {dep} failed to install (this is okay)")
    
    return True

def verify_installation():
    """Verify that the installation works"""
    print("🧪 Verifying installation...")
    
    # Determine python command
    if platform.system() == "Windows":
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "venv/bin/python"
    
    # Test basic imports
    test_script = '''
import torch
import transformers
import titans_pytorch
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ Transformers version: {transformers.__version__}")
print(f"✅ TITANS PyTorch available")

# Check device availability
if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("✅ Apple Metal (MPS) available")
else:
    print("✅ CPU backend available")

print("🎉 All core dependencies verified!")
'''
    
    # Write test script to temporary file
    with open("test_install.py", "w") as f:
        f.write(test_script)
    
    # Run test
    success = run_command(f"{python_cmd} test_install.py", "Testing installation")
    
    # Clean up
    os.remove("test_install.py")
    
    return success

def create_quick_start_script():
    """Create a quick start script"""
    print("📝 Creating quick start script...")
    
    quick_start = '''#!/usr/bin/env python3
"""
Quick start script for Ultra-Long Context TITANS + Qwen
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from titans_qwen_ultra_long import UltraLongTitansQwen

def main():
    print("🚀 Ultra-Long Context TITANS + Qwen Quick Start")
    print("=" * 60)
    
    # Initialize model with 512K context
    print("📥 Loading model (this may take a few minutes)...")
    model = UltraLongTitansQwen(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_context_length=524288,  # 512K tokens
        ultra_long_mode=True
    )
    
    # Test with a simple prompt
    prompt = "The future of artificial intelligence and ultra-long context processing will"
    
    print(f"📝 Generating with prompt: '{prompt}'")
    print("-" * 60)
    
    result = model.generate_ultra_long_context(
        prompt,
        max_new_tokens=200,
        temperature=0.7
    )
    
    print(f"✨ Generated text:")
    print(result["generated_text"])
    print(f"\\n📊 Stats:")
    print(f"Input length: {result['input_length']:,} tokens")
    print(f"Output length: {result['output_length']:,} tokens")
    print(f"Generation time: {result['generation_time']:.2f}s")
    print(f"Speed: {result['tokens_per_second']:.2f} tokens/sec")
    print(f"Used TITANS: {result['used_titans']}")
    print(f"Memory usage: {result['memory_usage']['process_rss']:.1f}GB")

if __name__ == "__main__":
    main()
'''
    
    with open("quick_start.py", "w") as f:
        f.write(quick_start)
    
    print("✅ Created quick_start.py")

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "="*80)
    print("🎉 SETUP COMPLETE!")
    print("="*80)
    print("\n📋 Next Steps:")
    print("\n1. Activate the virtual environment:")
    
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Test basic functionality:")
    print("   python test_titans.py")
    
    print("\n3. Run ultra-long context tests:")
    print("   python test_ultra_long_context.py --quick")
    
    print("\n4. Try the quick start demo:")
    print("   python quick_start.py")
    
    print("\n5. Run full 512K benchmark:")
    print("   python titans_qwen_ultra_long.py --benchmark --max-context 524288")
    
    print("\n6. Generate with ultra-long context:")
    print("   python titans_qwen_ultra_long.py --prompt 'Your prompt here' --max-tokens 500")
    
    print("\n🎯 For 512K token contexts, ensure you have:")
    print("   - 64GB+ RAM (recommended)")
    print("   - M3 Max or equivalent GPU")
    print("   - Sufficient disk space for model downloads")
    
    print("\n📚 Files created:")
    print("   - titans_qwen_ultra_long.py (main ultra-long implementation)")
    print("   - test_ultra_long_context.py (comprehensive test suite)")
    print("   - quick_start.py (quick demo script)")
    
    print("\n" + "="*80)

def main():
    """Main setup function"""
    print("🚀 Ultra-Long Context TITANS + Qwen Setup")
    print("=" * 60)
    print("Setting up environment for 512K token context windows")
    print("Optimized for M3 Max with 64GB RAM")
    print("=" * 60)
    
    # Check requirements
    if not check_python_version():
        print("❌ Setup failed: Python 3.11+ required")
        return False
    
    check_system_requirements()
    
    # Setup environment
    if not setup_virtual_environment():
        print("❌ Setup failed: Could not create virtual environment")
        return False
    
    if not install_dependencies():
        print("❌ Setup failed: Could not install dependencies")
        return False
    
    if not verify_installation():
        print("❌ Setup failed: Installation verification failed")
        return False
    
    # Create additional files
    create_quick_start_script()
    
    # Print instructions
    print_usage_instructions()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
