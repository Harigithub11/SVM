# Phase 1: Setup and Project Structure

## Overview
Set up the development environment, install all required dependencies, and create the project folder structure.

**Estimated Duration:** 1-2 hours

---

## 1. Environment Setup

### 1.1 Python Environment

#### Requirements:
- Python 3.8 or higher (3.9+ recommended)
- pip package manager
- Command-line access

#### Installation Commands:
```bash
# Check Python version
python --version

# Check pip version
pip --version
```

#### Expected Output:
- Python 3.8+ installed
- pip 20.0+ installed

---

### 1.2 Virtual Environment Setup

#### Why Virtual Environment?
- Isolates project dependencies
- Prevents version conflicts
- Makes project portable

#### Create Virtual Environment:
```bash
# Navigate to project directory
cd "C:\Hari\SRM\7th Sem\QML\Classic SVM"

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Activate virtual environment (Mac/Linux)
source venv/bin/activate
```

#### Verification:
```bash
# Check if virtual environment is active (should show venv path)
which python
```

---

### 1.3 IDE Configuration (Optional)

#### Recommended IDEs:
- **VS Code** - Lightweight, excellent Python support
- **PyCharm** - Full-featured Python IDE
- **Jupyter Notebook** - Interactive development

#### VS Code Setup:
1. Install Python extension
2. Select Python interpreter from venv
3. Enable auto-save
4. Install Pylint for code quality

---

## 2. Dependencies Installation

### 2.1 Core ML Libraries

#### Libraries to Install:
- **scikit-learn** - SVM implementation
- **numpy** - Numerical computations
- **pandas** - Data manipulation

#### Installation:
```bash
# Updated versions for 2024-2025 compatibility
pip install scikit-learn==1.3.2  # Includes HalvingGridSearchCV, ColumnTransformer improvements
pip install numpy==1.24.3
pip install pandas==2.0.3
```

#### Purpose:
- `scikit-learn`: Provides SVC (Support Vector Classification)
- `numpy`: Array operations, mathematical functions
- `pandas`: CSV handling, data preprocessing

---

### 2.2 Visualization Libraries

#### Libraries to Install:
- **matplotlib** - Basic plotting
- **seaborn** - Statistical visualizations
- **plotly** - Interactive charts

#### Installation:
```bash
# Updated versions for 2024-2025 compatibility
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install plotly==5.17.0
```

#### Purpose:
- `matplotlib`: Decision boundaries, basic plots
- `seaborn`: Confusion matrix, heatmaps
- `plotly`: Interactive 3D plots, ROC curves

---

### 2.3 Web Framework

#### Library to Install:
- **streamlit** - Web app framework

#### Installation:
```bash
# Streamlit 1.28+ includes st.status() for better progress tracking
pip install streamlit==1.28.0
```

#### Purpose:
- Creates interactive web interface
- Handles file uploads
- Manages UI components

---

### 2.4 Utility Libraries

#### Libraries to Install:
- **imbalanced-learn** - Handling imbalanced datasets (optional)
- **pillow** - Image handling for icons

#### Installation:
```bash
pip install imbalanced-learn==0.11.0
pip install pillow==10.0.1
pip install orjson==3.9.10  # Optional: Faster JSON parsing for Plotly (performance boost)
```

---

### 2.5 Complete Installation Command

#### Single Command Install:
```bash
# All dependencies with 2024-2025 verified versions
pip install scikit-learn==1.3.2 numpy==1.24.3 pandas==2.0.3 matplotlib==3.7.2 seaborn==0.12.2 plotly==5.17.0 streamlit==1.28.0 imbalanced-learn==0.11.0 pillow==10.0.1 orjson==3.9.10
```

---

## 3. Project Folder Structure

### 3.1 Complete Directory Structure

```
Classic SVM/
│
├── venv/                          # Virtual environment (auto-generated)
│
├── docs/                          # Documentation
│   ├── project_features.md
│   ├── project_phases_overview.md
│   ├── phase1_setup_and_structure.md
│   ├── phase2_data_and_preprocessing.md
│   ├── phase3_svm_implementation.md
│   ├── phase4_visualizations.md
│   └── phase5_streamlit_ui.md
│
├── data/                          # Datasets
│   ├── samples/                   # Sample datasets
│   │   ├── medical_sample.csv
│   │   ├── fraud_sample.csv
│   │   └── classification_sample.csv
│   └── uploaded/                  # User uploaded datasets (runtime)
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── app.py                     # Main Streamlit app
│   ├── data_handler.py            # Data loading & preprocessing
│   ├── svm_models.py              # SVM implementations
│   ├── visualizations.py          # All visualization functions
│   └── utils.py                   # Utility functions
│
├── assets/                        # Static assets
│   ├── images/                    # Icons, logos
│   └── styles/                    # Custom CSS
│       └── style.css
│
├── tests/                         # Test files (optional)
│   ├── test_data_handler.py
│   ├── test_svm_models.py
│   └── test_visualizations.py
│
├── requirements.txt               # Dependency list
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore file
└── config.py                      # Configuration settings (optional)
```

---

### 3.2 Create Directory Structure

#### Commands:
```bash
# Create main directories
mkdir data
mkdir data\samples
mkdir data\uploaded
mkdir src
mkdir assets
mkdir assets\images
mkdir assets\styles
mkdir tests

# Create __init__.py for Python package
type nul > src\__init__.py
```

---

## 4. Configuration Files

### 4.1 requirements.txt

#### Create File:
```bash
# In project root directory
```

#### Content:
```
# Core ML Libraries (2024-2025 verified versions)
scikit-learn==1.3.2  # Includes HalvingGridSearchCV, ColumnTransformer improvements
numpy==1.24.3
pandas==2.0.3

# Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0

# Web Framework
streamlit==1.28.0  # Includes st.status() for better UX

# Utilities
imbalanced-learn==0.11.0
pillow==10.0.1

# Performance (Optional but recommended)
orjson==3.9.10  # Faster JSON for Plotly
```

#### Purpose:
- Easy dependency installation: `pip install -r requirements.txt`
- Version tracking
- Deployment compatibility

---

### 4.2 README.md

#### Basic Structure:
```markdown
# Classic SVM Multi-Domain Application

## Description
Interactive Streamlit application for Support Vector Machine classification across multiple domains.

## Features
- 3 Domain options: Medical, Fraud Detection, Classification
- 3 Kernel types: Linear, RBF, Polynomial
- Automated preprocessing
- Comprehensive visualizations

## Installation
1. Clone repository
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`

## Usage
```bash
streamlit run src/app.py
```

## Requirements
- Python 3.8+
- See requirements.txt for dependencies

## Project Structure
[See folder structure above]

## Author
[Your Name]

## License
[Your License]
```

---

### 4.3 .gitignore

#### Content:
```
# Virtual Environment
venv/
env/
ENV/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Jupyter Notebook
.ipynb_checkpoints

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data (uploaded files)
data/uploaded/*
!data/uploaded/.gitkeep

# Streamlit
.streamlit/

# Logs
*.log
```

---

### 4.4 config.py (Optional)

#### Purpose:
Centralize configuration settings

#### Example Content:
```python
# config.py

# Application Settings
APP_TITLE = "Classic SVM Multi-Domain Application"
APP_ICON = "🤖"
PAGE_LAYOUT = "wide"

# Model Parameters
DEFAULT_C = 1.0
DEFAULT_GAMMA = 'scale'
DEFAULT_POLY_DEGREE = 3
RANDOM_STATE = 42

# Data Settings
TRAIN_TEST_SPLIT = 0.3
MIN_SAMPLES = 50
MAX_FILE_SIZE_MB = 10

# Visualization Settings
PLOT_DPI = 100
FIGURE_SIZE = (10, 6)
COLOR_PALETTE = 'viridis'

# Domain Configurations
DOMAINS = {
    'medical': {
        'name': 'Medical Diagnosis',
        'description': 'Disease prediction and medical classification',
        'sample_file': 'data/samples/medical_sample.csv'
    },
    'fraud': {
        'name': 'Fraud Detection',
        'description': 'Transaction fraud identification',
        'sample_file': 'data/samples/fraud_sample.csv'
    },
    'classification': {
        'name': 'General Classification',
        'description': 'Pattern recognition and data classification',
        'sample_file': 'data/samples/classification_sample.csv'
    }
}

# Kernel Configurations
KERNELS = {
    'linear': {
        'name': 'Linear Kernel',
        'description': 'For linearly separable data',
        'formula': 'K(x,y) = x·y'
    },
    'rbf': {
        'name': 'RBF Kernel',
        'description': 'For non-linear boundaries',
        'formula': 'K(x,y) = exp(-γ||x-y||²)'
    },
    'poly': {
        'name': 'Polynomial Kernel',
        'description': 'For curved decision boundaries',
        'formula': 'K(x,y) = (x·y + c)^d'
    }
}
```

---

## 5. Phase 1 Tasks & Subtasks

### Task 1.1: Environment Preparation
- [ ] Install Python 3.8+
- [ ] Install pip
- [ ] Verify installations

### Task 1.2: Virtual Environment
- [ ] Create virtual environment
- [ ] Activate virtual environment
- [ ] Verify activation

### Task 1.3: Install Dependencies
- [ ] Install scikit-learn
- [ ] Install numpy, pandas
- [ ] Install matplotlib, seaborn, plotly
- [ ] Install streamlit
- [ ] Install utility libraries
- [ ] Verify all installations

### Task 1.4: Create Folder Structure
- [ ] Create `data/` directory and subdirectories
- [ ] Create `src/` directory
- [ ] Create `assets/` directory and subdirectories
- [ ] Create `docs/` directory
- [ ] Create `tests/` directory
- [ ] Create `__init__.py` in src/

### Task 1.5: Configuration Files
- [ ] Create `requirements.txt`
- [ ] Create `README.md`
- [ ] Create `.gitignore`
- [ ] Create `config.py` (optional)

### Task 1.6: Verify Setup
- [ ] Check folder structure
- [ ] Run `pip list` to verify packages
- [ ] Test streamlit: `streamlit hello`

---

## 6. Testing Phase 1 Completion

### Test 1: Python Environment
```bash
python --version
# Expected: Python 3.8.x or higher
```

### Test 2: Virtual Environment
```bash
which python  # Mac/Linux
where python  # Windows
# Expected: Path should include 'venv'
```

### Test 3: Dependencies
```bash
pip list
# Expected: All packages from requirements.txt listed
```

### Test 4: Streamlit
```bash
streamlit --version
# Expected: Version 1.28.0 or compatible
```

### Test 5: Folder Structure
```bash
dir /B  # Windows
ls     # Mac/Linux
# Expected: data, src, assets, docs folders exist
```

### Test 6: Import Test
```bash
python -c "import sklearn, numpy, pandas, matplotlib, seaborn, plotly, streamlit; print('All imports successful')"
# Expected: "All imports successful"
```

---

## 7. Common Issues & Solutions

### Issue 1: Python not found
**Solution:** Add Python to PATH environment variable

### Issue 2: pip not found
**Solution:**
```bash
python -m ensurepip --upgrade
```

### Issue 3: Virtual environment activation fails
**Solution (Windows PowerShell):**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue 4: Package installation fails
**Solution:**
```bash
pip install --upgrade pip
pip cache purge
# Retry installation
```

### Issue 5: Streamlit won't run
**Solution:**
```bash
pip uninstall streamlit
pip install streamlit==1.28.0
```

---

## 8. Phase 1 Completion Checklist

### Environment ✓
- [ ] Python 3.8+ installed and verified
- [ ] Virtual environment created and active
- [ ] Command line accessible

### Dependencies ✓
- [ ] All required packages installed
- [ ] requirements.txt created
- [ ] No installation errors

### Structure ✓
- [ ] All folders created
- [ ] __init__.py in src/
- [ ] Folder structure matches specification

### Configuration ✓
- [ ] README.md created
- [ ] .gitignore created
- [ ] config.py created (optional)

### Verification ✓
- [ ] All tests passed
- [ ] Streamlit test successful
- [ ] Import test successful

---

## 9. Next Steps

Once Phase 1 is complete:
1. Proceed to **Phase 2: Data and Preprocessing**
2. Create sample datasets
3. Implement data loading functions

---

## 10. Time Tracking

**Estimated Time:** 1-2 hours
**Breakdown:**
- Environment setup: 20 minutes
- Dependencies installation: 15 minutes
- Folder structure creation: 10 minutes
- Configuration files: 15 minutes
- Testing & verification: 20 minutes
- Buffer for issues: 20-40 minutes

---

## Phase 1 Sign-Off

**Completed By:** ___________________
**Date:** ___________________
**Time Taken:** ___________________
**Issues Encountered:** ___________________
**Ready for Phase 2:** [ ] Yes [ ] No
