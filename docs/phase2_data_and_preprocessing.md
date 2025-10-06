# Phase 2: Data and Preprocessing

## Overview
Create sample datasets for all three domains, implement data loading functions, auto-detection modules, and preprocessing pipeline.

**Estimated Duration:** 3-4 hours

**Prerequisites:** Phase 1 completed

---

## 1. Sample Dataset Creation

### 1.1 Medical Domain Dataset

#### Dataset Specifications:
- **Purpose:** Disease diagnosis/prediction
- **Filename:** `medical_sample.csv`
- **Location:** `data/samples/`
- **Rows:** 200-300 samples
- **Features:** 8-10 medical indicators

#### Feature Examples:
- `age` (20-80)
- `blood_pressure` (80-180)
- `cholesterol` (150-300)
- `glucose` (70-200)
- `bmi` (15-40)
- `heart_rate` (60-120)
- `smoking` (0/1)
- `exercise` (0/1)
- `family_history` (0/1)
- **Target:** `disease` (0 = Healthy, 1 = Diseased)

#### Python Script to Generate:
```python
import pandas as pd
import numpy as np

np.random.seed(42)

# Generate medical data with realistic correlations (2024 best practice)
n_samples = 250

age = np.random.randint(20, 81, n_samples)

# Blood pressure correlates with age (more realistic)
blood_pressure = 80 + (age - 20) * 0.8 + np.random.normal(0, 10, n_samples)
blood_pressure = np.clip(blood_pressure, 80, 200).astype(int)

# Cholesterol also correlates with age
cholesterol = 150 + (age - 20) * 1.2 + np.random.normal(0, 15, n_samples)
cholesterol = np.clip(cholesterol, 150, 300).astype(int)

glucose = np.random.randint(70, 201, n_samples)

# BMI has weak correlation with age
bmi = 22 + (age - 50) * 0.05 + np.random.normal(0, 3, n_samples)
bmi = np.clip(bmi, 15, 40).round(1)
heart_rate = np.random.randint(60, 121, n_samples)
smoking = np.random.randint(0, 2, n_samples)
exercise = np.random.randint(0, 2, n_samples)
family_history = np.random.randint(0, 2, n_samples)

# Create target with logic
disease = np.zeros(n_samples)
for i in range(n_samples):
    risk_score = (
        (age[i] > 60) * 2 +
        (blood_pressure[i] > 140) * 2 +
        (cholesterol[i] > 240) * 2 +
        (glucose[i] > 140) * 2 +
        (bmi[i] > 30) * 1 +
        smoking[i] * 2 +
        (exercise[i] == 0) * 1 +
        family_history[i] * 2
    )
    disease[i] = 1 if risk_score > 6 else 0

# Add some noise
noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
disease[noise_indices] = 1 - disease[noise_indices]

# Create DataFrame
medical_df = pd.DataFrame({
    'age': age,
    'blood_pressure': blood_pressure,
    'cholesterol': cholesterol,
    'glucose': glucose,
    'bmi': bmi,
    'heart_rate': heart_rate,
    'smoking': smoking,
    'exercise': exercise,
    'family_history': family_history,
    'disease': disease.astype(int)
})

# Save
medical_df.to_csv('data/samples/medical_sample.csv', index=False)
print("Medical dataset created:", medical_df.shape)
print(medical_df['disease'].value_counts())
```

---

### 1.2 Fraud Detection Dataset

#### Dataset Specifications:
- **Purpose:** Transaction fraud detection
- **Filename:** `fraud_sample.csv`
- **Location:** `data/samples/`
- **Rows:** 300-400 samples
- **Features:** 8-10 transaction features

#### Feature Examples:
- `transaction_amount` (10-5000)
- `transaction_hour` (0-23)
- `day_of_week` (1-7)
- `merchant_category` (1-10)
- `card_type` (0-2)
- `location_distance` (0-1000 km)
- `num_transactions_day` (1-20)
- `avg_transaction_amount` (50-1000)
- `account_age_days` (30-3650)
- **Target:** `fraud` (0 = Legitimate, 1 = Fraud)

#### Python Script to Generate:
```python
import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 350

transaction_amount = np.random.uniform(10, 5000, n_samples).round(2)
transaction_hour = np.random.randint(0, 24, n_samples)
day_of_week = np.random.randint(1, 8, n_samples)
merchant_category = np.random.randint(1, 11, n_samples)
card_type = np.random.randint(0, 3, n_samples)
location_distance = np.random.uniform(0, 1000, n_samples).round(2)
num_transactions_day = np.random.randint(1, 21, n_samples)
avg_transaction_amount = np.random.uniform(50, 1000, n_samples).round(2)
account_age_days = np.random.randint(30, 3651, n_samples)

# Create fraud target with logic
fraud = np.zeros(n_samples)
for i in range(n_samples):
    fraud_score = (
        (transaction_amount[i] > 2000) * 2 +
        (transaction_hour[i] < 6 or transaction_hour[i] > 22) * 2 +
        (location_distance[i] > 500) * 2 +
        (num_transactions_day[i] > 10) * 2 +
        (account_age_days[i] < 180) * 1 +
        (transaction_amount[i] > avg_transaction_amount[i] * 3) * 3
    )
    fraud[i] = 1 if fraud_score > 5 else 0

# Add noise
noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.08), replace=False)
fraud[noise_indices] = 1 - fraud[noise_indices]

fraud_df = pd.DataFrame({
    'transaction_amount': transaction_amount,
    'transaction_hour': transaction_hour,
    'day_of_week': day_of_week,
    'merchant_category': merchant_category,
    'card_type': card_type,
    'location_distance': location_distance,
    'num_transactions_day': num_transactions_day,
    'avg_transaction_amount': avg_transaction_amount,
    'account_age_days': account_age_days,
    'fraud': fraud.astype(int)
})

fraud_df.to_csv('data/samples/fraud_sample.csv', index=False)
print("Fraud dataset created:", fraud_df.shape)
print(fraud_df['fraud'].value_counts())
```

---

### 1.3 General Classification Dataset

#### Dataset Specifications:
- **Purpose:** Multi-class pattern classification
- **Filename:** `classification_sample.csv`
- **Location:** `data/samples/`
- **Rows:** 300-400 samples
- **Features:** 6-8 features
- **Classes:** 3 classes (0, 1, 2)

#### Feature Examples:
- `feature_1` to `feature_8` (numerical)
- **Target:** `class` (0, 1, or 2)

#### Python Script to Generate:
```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

np.random.seed(42)

# Generate synthetic classification dataset with difficulty control (2024)
X, y = make_classification(
    n_samples=350,
    n_features=8,
    n_informative=6,
    n_redundant=2,
    n_classes=3,
    n_clusters_per_class=1,
    class_sep=1.0,         # NEW: Controls class separability (higher = easier)
    flip_y=0.05,          # Label noise for realism
    weights=[0.33, 0.33, 0.34],
    random_state=42        # CRITICAL: Always set for reproducibility
)

# Create DataFrame
classification_df = pd.DataFrame(
    X,
    columns=[f'feature_{i+1}' for i in range(8)]
)
classification_df['class'] = y

classification_df.to_csv('data/samples/classification_sample.csv', index=False)
print("Classification dataset created:", classification_df.shape)
print(classification_df['class'].value_counts())
```

---

### 1.4 Dataset Validation

#### Validation Script:
```python
import pandas as pd
import os

def validate_dataset(filepath):
    """Validate dataset meets requirements"""
    if not os.path.exists(filepath):
        print(f"❌ {filepath} not found")
        return False

    df = pd.read_csv(filepath)
    print(f"✓ {filepath}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Target distribution:\n{df.iloc[:, -1].value_counts()}\n")
    return True

# Validate all datasets
validate_dataset('data/samples/medical_sample.csv')
validate_dataset('data/samples/fraud_sample.csv')
validate_dataset('data/samples/classification_sample.csv')
```

---

## 2. Data Loading Module

### 2.1 CSV Reader Function

#### File: `src/data_handler.py`

#### Function: `load_dataset()`
```python
import pandas as pd
import os

def load_dataset(filepath):
    """
    Load dataset from CSV file

    Args:
        filepath (str): Path to CSV file

    Returns:
        pd.DataFrame: Loaded dataset

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        df = pd.read_csv(filepath)

        if df.empty:
            raise ValueError("Dataset is empty")

        if df.shape[0] < 10:
            raise ValueError("Dataset too small (minimum 10 rows required)")

        return df

    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty")
    except pd.errors.ParserError:
        raise ValueError("Invalid CSV format")
```

---

### 2.2 Dataset Validator

#### Function: `validate_dataset()`
```python
def validate_dataset(df, min_samples=50, max_file_size_mb=10):
    """
    Validate dataset meets requirements

    Args:
        df (pd.DataFrame): Dataset to validate
        min_samples (int): Minimum number of samples
        max_file_size_mb (int): Maximum file size in MB

    Returns:
        dict: Validation results
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

    # Check minimum samples
    if len(df) < min_samples:
        validation_results['errors'].append(
            f"Dataset has {len(df)} rows, minimum {min_samples} required"
        )
        validation_results['valid'] = False

    # Check for all NaN columns
    nan_cols = df.columns[df.isnull().all()].tolist()
    if nan_cols:
        validation_results['errors'].append(
            f"Columns with all NaN values: {nan_cols}"
        )
        validation_results['valid'] = False

    # Check for single value columns
    single_value_cols = [col for col in df.columns if df[col].nunique() == 1]
    if single_value_cols:
        validation_results['warnings'].append(
            f"Columns with single unique value: {single_value_cols}"
        )

    # Check for missing values
    missing_percent = (df.isnull().sum() / len(df) * 100).round(2)
    high_missing = missing_percent[missing_percent > 50].to_dict()
    if high_missing:
        validation_results['warnings'].append(
            f"Columns with >50% missing values: {high_missing}"
        )

    return validation_results
```

---

### 2.3 Sample Dataset Loader

#### Function: `load_sample_dataset()`
```python
def load_sample_dataset(domain):
    """
    Load sample dataset for specified domain

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
        raise ValueError(f"Invalid domain: {domain}")

    filepath = sample_files[domain]
    return load_dataset(filepath)
```

---

## 3. Auto-Detection Module

### 3.1 Feature Type Detection

#### Function: `detect_feature_types()`
```python
import numpy as np

def detect_feature_types(df):
    """
    Automatically detect numerical and categorical features

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        dict: Feature types classification
    """
    feature_types = {
        'numerical': [],
        'categorical': [],
        'binary': [],
        'datetime': []
    }

    for col in df.columns:
        # Skip if all NaN
        if df[col].isnull().all():
            continue

        # Check datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            feature_types['datetime'].append(col)

        # Check numerical
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Check if binary (only 2 unique values)
            if df[col].nunique() == 2:
                feature_types['binary'].append(col)
            else:
                feature_types['numerical'].append(col)

        # Categorical
        else:
            # Convert object to categorical
            unique_count = df[col].nunique()
            if unique_count < len(df) * 0.5:  # Less than 50% unique
                feature_types['categorical'].append(col)
            else:
                # Too many unique values, might be ID
                feature_types['categorical'].append(col)

    return feature_types
```

---

### 3.2 Target Column Detection

#### Function: `detect_target_column()`
```python
def detect_target_column(df, possible_names=None):
    """
    Detect target column in dataset

    Args:
        df (pd.DataFrame): Input dataset
        possible_names (list): Possible target column names

    Returns:
        str: Target column name
    """
    if possible_names is None:
        possible_names = [
            'target', 'label', 'class', 'output', 'y',
            'disease', 'fraud', 'diagnosis', 'result'
        ]

    # Check for exact matches
    for name in possible_names:
        if name in df.columns:
            return name

    # Check for partial matches
    for col in df.columns:
        for name in possible_names:
            if name in col.lower():
                return col

    # Default to last column
    return df.columns[-1]
```

---

### 3.3 Missing Value Detection

#### Function: `detect_missing_values()`
```python
def detect_missing_values(df):
    """
    Detect and analyze missing values

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        dict: Missing value analysis
    """
    missing_info = {
        'total_missing': df.isnull().sum().sum(),
        'missing_by_column': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'rows_with_missing': df.isnull().any(axis=1).sum()
    }

    # Identify columns with missing values
    missing_info['columns_with_missing'] = [
        col for col in df.columns if df[col].isnull().any()
    ]

    return missing_info
```

---

### 3.4 Data Quality Checks

#### Function: `check_data_quality()`
```python
def check_data_quality(df):
    """
    Perform comprehensive data quality checks

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        dict: Quality check results
    """
    quality_report = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'duplicates': df.duplicated().sum(),
        'data_types': df.dtypes.value_counts().to_dict()
    }

    # Check for outliers in numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outlier_count > 0:
            outliers[col] = outlier_count

    quality_report['outliers'] = outliers

    return quality_report
```

---

## 4. Preprocessing Module

### 4.1 Missing Value Handler

#### Function: `handle_missing_values()`
```python
from sklearn.impute import SimpleImputer

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in dataset

    Args:
        df (pd.DataFrame): Input dataset
        strategy (str): Imputation strategy ('mean', 'median', 'most_frequent')

    Returns:
        pd.DataFrame: Dataset with imputed values
    """
    df_copy = df.copy()

    # Separate numerical and categorical columns
    numerical_cols = df_copy.select_dtypes(include=[np.number]).columns
    categorical_cols = df_copy.select_dtypes(include=['object']).columns

    # Impute numerical columns
    if len(numerical_cols) > 0 and df_copy[numerical_cols].isnull().any().any():
        num_imputer = SimpleImputer(strategy=strategy)
        df_copy[numerical_cols] = num_imputer.fit_transform(df_copy[numerical_cols])

    # Impute categorical columns
    if len(categorical_cols) > 0 and df_copy[categorical_cols].isnull().any().any():
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_copy[categorical_cols] = cat_imputer.fit_transform(df_copy[categorical_cols])

    return df_copy
```

---

### 4.2 Categorical Encoder

#### Function: `encode_categorical_features()`
```python
from sklearn.preprocessing import LabelEncoder

def encode_categorical_features(df, target_column):
    """
    Encode categorical features to numerical

    Args:
        df (pd.DataFrame): Input dataset
        target_column (str): Target column name

    Returns:
        tuple: (encoded_df, encoders_dict)
    """
    df_copy = df.copy()
    encoders = {}

    # Get categorical columns (exclude target)
    categorical_cols = df_copy.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != target_column]

    # Encode each categorical column
    for col in categorical_cols:
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(df_copy[col].astype(str))
        encoders[col] = le

    return df_copy, encoders
```

---

### 4.3 Feature Scaler

#### Function: `scale_features()`
```python
from sklearn.preprocessing import StandardScaler

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler

    Args:
        X_train (array-like): Training features
        X_test (array-like): Testing features

    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
```

---

### 4.4 Train-Test Splitter

#### Function: `split_data()`
```python
from sklearn.model_selection import train_test_split

def split_data(df, target_column, test_size=0.3, random_state=42):
    """
    Split dataset into train and test sets

    Args:
        df (pd.DataFrame): Input dataset
        target_column (str): Target column name
        test_size (float): Test set proportion
        random_state (int): Random seed

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(y.unique()) > 1 else None
    )

    return X_train, X_test, y_train, y_test
```

---

### 4.5 Complete Preprocessing Pipeline

#### Function: `preprocess_pipeline()`
```python
def preprocess_pipeline(df, target_column=None, test_size=0.3):
    """
    Complete preprocessing pipeline

    Args:
        df (pd.DataFrame): Input dataset
        target_column (str): Target column name (auto-detected if None)
        test_size (float): Test set proportion

    Returns:
        dict: Preprocessing results
    """
    # Auto-detect target if not provided
    if target_column is None:
        target_column = detect_target_column(df)

    # Handle missing values
    df_clean = handle_missing_values(df)

    # Encode categorical features
    df_encoded, encoders = encode_categorical_features(df_clean, target_column)

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        df_encoded, target_column, test_size
    )

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'encoders': encoders,
        'target_column': target_column,
        'feature_names': X_train.columns.tolist()
    }
```

---

### 4.6 Modern Preprocessing with ColumnTransformer (2024 - RECOMMENDED)

#### ⚠️ IMPORTANT: Prevents Data Leakage!

The `preprocess_pipeline()` above works, but can cause **data leakage** if not careful. The modern approach using `ColumnTransformer` is **safer** and **industry-standard**.

#### Why ColumnTransformer?
- **Prevents data leakage** - fit only on training data
- Handles mixed data types (numeric + categorical)
- Feature names preserved (NEW in sklearn 1.7.2)
- Cleaner, more maintainable code

#### Function: `modern_preprocessing_pipeline()`
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def modern_preprocessing_pipeline(df, target_column=None, test_size=0.3):
    """
    Modern preprocessing using ColumnTransformer (RECOMMENDED 2024)

    Advantages over manual approach:
    - No data leakage (fit only on train)
    - Proper handling of mixed types
    - Feature names preserved
    - Production-ready

    Args:
        df (pd.DataFrame): Input dataset
        target_column (str): Target column (auto-detected if None)
        test_size (float): Test set proportion

    Returns:
        dict: Preprocessing results
    """
    # Auto-detect target
    if target_column is None:
        target_column = detect_target_column(df)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify column types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Numeric transformer pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Median more robust than mean
        ('scaler', StandardScaler())
    ])

    # Categorical transformer pipeline
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
        verbose_feature_names_out=False  # NEW in sklearn 1.7 - cleaner names
    )

    # Stratified split (important for imbalanced data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None  # Prevents class imbalance issues
    )

    # Fit preprocessor ONLY on training data (prevents leakage!)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)  # Only transform, not fit!

    # Get feature names (NEW in sklearn 1.7)
    feature_names = preprocessor.get_feature_names_out()

    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'preprocessor': preprocessor,
        'feature_names': feature_names.tolist(),
        'target_column': target_column,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }
```

#### Usage Example:
```python
# Load data
df = load_sample_dataset('medical')

# Modern preprocessing (RECOMMENDED)
data = modern_preprocessing_pipeline(df)

# Access preprocessed data
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
```

#### Comparison: Old vs New

| Aspect | `preprocess_pipeline()` | `modern_preprocessing_pipeline()` |
|--------|------------------------|-----------------------------------|
| **Data Leakage Risk** | ⚠️ Possible if misused | ✅ Prevented |
| **Mixed Types** | ⚠️ Manual handling | ✅ Automatic |
| **Feature Names** | ❌ Lost after encoding | ✅ Preserved |
| **Industry Standard** | ⚠️ Basic | ✅ Production-ready |
| **Code Complexity** | Higher | Lower |

**Recommendation:** Use `modern_preprocessing_pipeline()` for production code. Keep `preprocess_pipeline()` for learning/simple cases.

---

## 5. Phase 2 Tasks & Subtasks

### Task 2.1: Create Sample Datasets
- [ ] Generate medical_sample.csv
- [ ] Generate fraud_sample.csv
- [ ] Generate classification_sample.csv
- [ ] Validate all datasets
- [ ] Verify data quality

### Task 2.2: Implement Data Loading
- [ ] Create `load_dataset()` function
- [ ] Implement error handling
- [ ] Create `validate_dataset()` function
- [ ] Create `load_sample_dataset()` function
- [ ] Test with all sample datasets

### Task 2.3: Implement Auto-Detection
- [ ] Create `detect_feature_types()` function
- [ ] Create `detect_target_column()` function
- [ ] Create `detect_missing_values()` function
- [ ] Create `check_data_quality()` function
- [ ] Test auto-detection on all datasets

### Task 2.4: Implement Preprocessing
- [ ] Create `handle_missing_values()` function
- [ ] Create `encode_categorical_features()` function
- [ ] Create `scale_features()` function
- [ ] Create `split_data()` function
- [ ] Create `preprocess_pipeline()` function

### Task 2.5: Integration Testing
- [ ] Test complete pipeline on medical data
- [ ] Test complete pipeline on fraud data
- [ ] Test complete pipeline on classification data
- [ ] Verify all outputs are correct
- [ ] Handle edge cases

---

## 6. Testing Phase 2 Completion

### Test 1: Dataset Creation
```python
import os
import pandas as pd

# Check all datasets exist
datasets = [
    'data/samples/medical_sample.csv',
    'data/samples/fraud_sample.csv',
    'data/samples/classification_sample.csv'
]

for dataset in datasets:
    assert os.path.exists(dataset), f"{dataset} not found"
    df = pd.read_csv(dataset)
    assert len(df) >= 200, f"{dataset} has insufficient rows"
    print(f"✓ {dataset}: {df.shape}")
```

### Test 2: Data Loading
```python
from src.data_handler import load_dataset, load_sample_dataset

# Test loading
df_medical = load_dataset('data/samples/medical_sample.csv')
assert df_medical is not None
assert len(df_medical) > 0

# Test sample loader
df_fraud = load_sample_dataset('fraud')
assert df_fraud is not None
print("✓ Data loading works")
```

### Test 3: Auto-Detection
```python
from src.data_handler import detect_feature_types, detect_target_column

df = load_sample_dataset('medical')
feature_types = detect_feature_types(df)
target = detect_target_column(df)

assert 'numerical' in feature_types
assert target is not None
print(f"✓ Auto-detection works. Target: {target}")
```

### Test 4: Preprocessing
```python
from src.data_handler import preprocess_pipeline

df = load_sample_dataset('classification')
results = preprocess_pipeline(df)

assert results['X_train'] is not None
assert results['X_test'] is not None
assert len(results['y_train']) > 0
print(f"✓ Preprocessing works. Train size: {len(results['X_train'])}")
```

### Test 5: Complete Pipeline
```python
# Test on all domains
for domain in ['medical', 'fraud', 'classification']:
    df = load_sample_dataset(domain)
    results = preprocess_pipeline(df)
    print(f"✓ {domain}: {results['X_train'].shape}, {results['X_test'].shape}")
```

---

## 7. Common Issues & Solutions

### Issue 1: Dataset generation fails
**Solution:** Install required packages, check numpy version

### Issue 2: CSV encoding errors
**Solution:** Use `pd.read_csv(filepath, encoding='utf-8')`

### Issue 3: Scaling fails with NaN
**Solution:** Ensure missing values handled before scaling

### Issue 4: Imbalanced classes
**Solution:** Use stratify in train_test_split

### Issue 5: Categorical encoding error
**Solution:** Convert to string before encoding: `df[col].astype(str)`

---

## 8. Phase 2 Completion Checklist

### Datasets ✓
- [ ] medical_sample.csv created and validated
- [ ] fraud_sample.csv created and validated
- [ ] classification_sample.csv created and validated
- [ ] All datasets have proper structure
- [ ] Target distributions are balanced

### Data Loading ✓
- [ ] load_dataset() implemented
- [ ] validate_dataset() implemented
- [ ] load_sample_dataset() implemented
- [ ] Error handling working
- [ ] All tests passed

### Auto-Detection ✓
- [ ] Feature type detection working
- [ ] Target detection working
- [ ] Missing value detection working
- [ ] Quality checks working

### Preprocessing ✓
- [ ] Missing value handling working
- [ ] Categorical encoding working
- [ ] Feature scaling working
- [ ] Train-test split working
- [ ] Complete pipeline working

### Testing ✓
- [ ] All unit tests passed
- [ ] Integration tests passed
- [ ] Edge cases handled
- [ ] No errors or warnings

---

## 9. Next Steps

Once Phase 2 is complete:
1. Proceed to **Phase 3: SVM Implementation**
2. Implement SVM models for all kernels
3. Create training and evaluation functions

---

## 10. Time Tracking

**Estimated Time:** 3-4 hours
**Breakdown:**
- Dataset creation: 45 minutes
- Data loading module: 30 minutes
- Auto-detection module: 45 minutes
- Preprocessing module: 1 hour
- Testing & debugging: 45 minutes
- Buffer: 15-45 minutes

---

## Phase 2 Sign-Off

**Completed By:** ___________________
**Date:** ___________________
**Time Taken:** ___________________
**Issues Encountered:** ___________________
**Ready for Phase 3:** [ ] Yes [ ] No
