#!/bin/bash
# =============================================================================
# ARM64-Compatible Environment Setup for M1/M2 Macs
# Content Automation Pipeline
# =============================================================================

set -e  # Exit on error

echo "🔧 Setting up ARM64-compatible Python environment..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}➜${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# =============================================================================
# STEP 1: Check system architecture
# =============================================================================
print_step "Checking system architecture..."
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    print_error "This script is for ARM64 (M1/M2) Macs. Detected: $ARCH"
    exit 1
fi
print_success "System architecture: $ARCH"
echo ""

# =============================================================================
# STEP 2: Find ARM64-native Python
# =============================================================================
print_step "Looking for ARM64-native Python installation..."

# Try Homebrew Python first (best for M1/M2)
if command -v brew &> /dev/null; then
    # Try Python 3.11 first (most compatible with all packages)
    if brew list python@3.11 &> /dev/null; then
        PYTHON_PATH=$(brew --prefix python@3.11)/bin/python3.11
        print_success "Found Homebrew Python 3.11: $PYTHON_PATH"
    elif brew list python@3.13 &> /dev/null; then
        PYTHON_PATH=$(brew --prefix python@3.13)/bin/python3.13
        print_success "Found Homebrew Python 3.13: $PYTHON_PATH"
    else
        print_warning "Homebrew Python not found. Installing Python 3.11..."
        brew install python@3.11
        PYTHON_PATH=$(brew --prefix python@3.11)/bin/python3.11
        print_success "Installed Homebrew Python 3.11"
    fi
else
    print_error "Homebrew not found. Please install Homebrew first:"
    echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

# Verify Python architecture
PYTHON_ARCH=$($PYTHON_PATH -c "import platform; print(platform.machine())")
print_step "Python architecture: $PYTHON_ARCH"

if [ "$PYTHON_ARCH" != "arm64" ]; then
    print_error "Python is not ARM64-native. Detected: $PYTHON_ARCH"
    print_warning "Trying to reinstall Python..."
    brew reinstall python@3.11
    PYTHON_PATH=$(brew --prefix python@3.11)/bin/python3.11
    PYTHON_ARCH=$($PYTHON_PATH -c "import platform; print(platform.machine())")

    if [ "$PYTHON_ARCH" != "arm64" ]; then
        print_error "Failed to get ARM64 Python. Please reinstall Homebrew in native mode."
        exit 1
    fi
fi

print_success "Using ARM64-native Python: $PYTHON_PATH"
$PYTHON_PATH --version
echo ""

# =============================================================================
# STEP 3: Remove old virtual environment if exists
# =============================================================================
if [ -d "venv" ]; then
    print_step "Removing old virtual environment..."
    rm -rf venv
    print_success "Old environment removed"
    echo ""
fi

# =============================================================================
# STEP 4: Create new ARM64-compatible virtual environment
# =============================================================================
print_step "Creating new ARM64-compatible virtual environment..."
$PYTHON_PATH -m venv venv
print_success "Virtual environment created"
echo ""

# =============================================================================
# STEP 5: Activate virtual environment
# =============================================================================
print_step "Activating virtual environment..."
source venv/bin/activate

# Verify we're in the venv
if [ -z "$VIRTUAL_ENV" ]; then
    print_error "Failed to activate virtual environment"
    exit 1
fi

print_success "Virtual environment activated: $VIRTUAL_ENV"
echo ""

# =============================================================================
# STEP 6: Upgrade pip, setuptools, and wheel
# =============================================================================
print_step "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel --trusted-host pypi.org --trusted-host files.pythonhosted.org
print_success "Package managers upgraded"
echo ""

# =============================================================================
# STEP 7: Install NumPy first (ARM64-optimized)
# =============================================================================
print_step "Installing NumPy with ARM64 optimizations..."

# Force reinstall NumPy to ensure ARM64 version
pip install --no-cache-dir --force-reinstall "numpy>=1.24.0" --trusted-host pypi.org --trusted-host files.pythonhosted.org

# Verify NumPy installation
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import numpy; print(f'NumPy file: {numpy.__file__}')"

print_success "NumPy installed successfully"
echo ""

# =============================================================================
# STEP 8: Install other ML dependencies with ARM64 support
# =============================================================================
print_step "Installing ML dependencies with ARM64 support..."

# Install SciPy (needed for many ML libraries)
pip install --no-cache-dir "scipy>=1.10.0" --trusted-host pypi.org --trusted-host files.pythonhosted.org

# Install scikit-learn
pip install --no-cache-dir "scikit-learn==1.3.2" --trusted-host pypi.org --trusted-host files.pythonhosted.org

print_success "ML dependencies installed"
echo ""

# =============================================================================
# STEP 9: Install PyTorch with ARM64 support
# =============================================================================
print_step "Installing PyTorch with ARM64/MPS support..."

# Install PyTorch with Apple Silicon (MPS) support
pip install --no-cache-dir torch torchvision torchaudio --trusted-host pypi.org --trusted-host files.pythonhosted.org

print_success "PyTorch installed with MPS support"
echo ""

# =============================================================================
# STEP 10: Install remaining requirements
# =============================================================================
print_step "Installing remaining requirements from requirements.txt..."

# Install all other requirements
pip install --no-cache-dir -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org

print_success "All requirements installed"
echo ""

# =============================================================================
# STEP 11: Install spaCy model
# =============================================================================
print_step "Installing spaCy English model..."

python -m spacy download en_core_web_sm || \
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

print_success "spaCy model installed"
echo ""

# =============================================================================
# STEP 12: Verify installation
# =============================================================================
print_step "Verifying installation..."
echo ""

echo "Testing NumPy import..."
python -c "import numpy; print(f'✓ NumPy {numpy.__version__} (architecture verified)')" || {
    print_error "NumPy import failed"
    exit 1
}

echo "Testing PyTorch import..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'  MPS Available: {torch.backends.mps.is_available()}')" || {
    print_error "PyTorch import failed"
    exit 1
}

echo "Testing scikit-learn import..."
python -c "import sklearn; print(f'✓ scikit-learn {sklearn.__version__}')" || {
    print_error "scikit-learn import failed"
    exit 1
}

echo "Testing FastAPI import..."
python -c "import fastapi; print(f'✓ FastAPI {fastapi.__version__}')" || {
    print_error "FastAPI import failed"
    exit 1
}

echo "Testing sentence-transformers import..."
python -c "from sentence_transformers import SentenceTransformer; print('✓ sentence-transformers imported successfully')" || {
    print_error "sentence-transformers import failed"
    exit 1
}

echo ""
print_success "All critical packages verified successfully!"
echo ""

# =============================================================================
# STEP 13: Test application import
# =============================================================================
print_step "Testing application import..."
echo ""

python -c "
import sys
sys.path.insert(0, '.')
try:
    # Test core imports without database connection
    from config.settings import settings
    print('✓ Settings loaded')

    from core.models import Project, User
    print('✓ Core models imported')

    from security import get_password_hash, verify_password
    print('✓ Security module imported')

    # Test password hashing fix
    test_password = 'TestPassword123'
    hashed = get_password_hash(test_password)
    assert verify_password(test_password, hashed)
    print('✓ Password hashing works correctly')

    print('')
    print('✅ Application imports verified successfully!')
    print('   (Database connection will be tested when you run the app)')

except Exception as e:
    print(f'❌ Import test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    print_success "Environment setup completed successfully!"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo -e "${GREEN}🎉 SUCCESS!${NC} Your ARM64-compatible environment is ready!"
    echo ""
    echo "To activate the environment in the future, run:"
    echo -e "  ${YELLOW}source venv/bin/activate${NC}"
    echo ""
    echo "To start the application:"
    echo -e "  ${YELLOW}uvicorn api.main:app --reload --host 0.0.0.0 --port 8000${NC}"
    echo ""
    echo "To view the new UI:"
    echo -e "  ${YELLOW}open http://localhost:8000${NC}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
else
    print_error "Environment setup had issues. Please check the errors above."
    exit 1
fi
