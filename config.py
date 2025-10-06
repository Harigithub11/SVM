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
