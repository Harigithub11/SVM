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
3. Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
4. Install dependencies: `pip install -r requirements.txt`

## Usage
```bash
streamlit run src/app.py
```

## Requirements
- Python 3.8+
- See requirements.txt for dependencies

## Project Structure
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
│
├── tests/                         # Test files (optional)
│
├── requirements.txt               # Dependency list
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore file
└── config.py                      # Configuration settings (optional)
```

## Author
[Your Name]

## License
[Your License]
