"""
Data Handler Module
Handles data loading, validation, and preprocessing with ColumnTransformer
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def load_dataset(file_path):
    """
    Load dataset from CSV file

    Args:
        file_path (str): Path to CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")

def load_sample_dataset(domain):
    """
    Load predefined sample dataset

    Args:
        domain (str): Domain name ('medical', 'fraud', 'classification')

    Returns:
        pd.DataFrame: Sample dataset
    """
    sample_files = {
        'medical': 'data/samples/medical_sample.csv',
        'fraud': 'data/samples/fraud_sample.csv',
        'classification': 'data/samples/classification_sample.csv'
    }

    if domain not in sample_files:
        raise ValueError(f"Unknown domain: {domain}")

    return load_dataset(sample_files[domain])

def validate_dataset(df, min_samples=50, min_features=2):
    """
    Validate dataset structure

    Args:
        df (pd.DataFrame): Dataset to validate
        min_samples (int): Minimum number of samples
        min_features (int): Minimum number of features

    Returns:
        dict: Validation result with errors list
    """
    errors = []

    if df.empty:
        errors.append("Dataset is empty")

    if df.shape[0] < min_samples:
        errors.append(f"Dataset too small: {df.shape[0]} rows (minimum {min_samples} required)")

    if df.shape[1] < min_features:
        errors.append(f"Insufficient columns: {df.shape[1]} (minimum {min_features} required)")

    # Check for columns with all missing values
    all_null_cols = df.columns[df.isnull().all()].tolist()
    if all_null_cols:
        errors.append(f"Columns contain only missing values: {', '.join(all_null_cols)}")

    return {'valid': len(errors) == 0, 'errors': errors}

def detect_feature_types(df, exclude_target=None):
    """
    Detect numeric and categorical feature types

    Args:
        df (pd.DataFrame): Dataset
        exclude_target (str): Target column name to exclude

    Returns:
        dict: Dictionary with numeric and categorical column lists
    """
    if exclude_target and exclude_target in df.columns:
        features_df = df.drop(columns=[exclude_target])
    else:
        features_df = df

    numeric_features = features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = features_df.select_dtypes(include=['object', 'category']).columns.tolist()

    return {
        'numeric': numeric_features,
        'categorical': categorical_features
    }

def detect_target_column(df):
    """
    Auto-detect target column (usually the last column or named 'target', 'label', etc.)

    Args:
        df (pd.DataFrame): Dataset

    Returns:
        str: Target column name
    """
    # Common target column names
    common_names = ['target', 'label', 'class', 'diagnosis', 'fraud_label', 'output', 'y']

    # Check for common names (case-insensitive)
    for col in df.columns:
        if col.lower() in common_names:
            return col

    # Default to last column
    return df.columns[-1]

def modern_preprocessing_pipeline(df, target_column=None, test_size=0.3, random_state=42):
    """
    Modern preprocessing using ColumnTransformer (RECOMMENDED 2024)
    Prevents data leakage by fitting only on training data

    Args:
        df (pd.DataFrame): Input dataset
        target_column (str): Target column name (auto-detect if None)
        test_size (float): Test set proportion
        random_state (int): Random seed

    Returns:
        dict: Preprocessed data dictionary
    """
    # Auto-detect target column if not provided
    if target_column is None:
        target_column = detect_target_column(df)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify column types
    feature_types = detect_feature_types(df, exclude_target=target_column)
    numeric_features = feature_types['numeric']
    categorical_features = feature_types['categorical']

    # Create transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        verbose_feature_names_out=False
    )

    # Split data BEFORE fitting (prevents data leakage!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit preprocessor ONLY on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after transformation
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        # Fallback for older sklearn versions
        feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])]

    # Encode target if categorical
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        class_names = le.classes_.tolist()
    else:
        class_names = np.unique(y).tolist()

    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names.tolist(),
        'class_names': class_names,
        'preprocessor': preprocessor,
        'original_X_train': X_train,
        'original_X_test': X_test
    }

def preprocess_pipeline(df, target_column=None, test_size=0.3):
    """
    Wrapper function for preprocessing pipeline

    Args:
        df (pd.DataFrame): Input dataset
        target_column (str): Target column name
        test_size (float): Test set proportion

    Returns:
        dict: Preprocessed data dictionary
    """
    return modern_preprocessing_pipeline(df, target_column, test_size)

def get_dataset_info(df):
    """
    Get comprehensive dataset information

    Args:
        df (pd.DataFrame): Dataset

    Returns:
        dict: Dataset information
    """
    return {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
        'duplicates': df.duplicated().sum()
    }
