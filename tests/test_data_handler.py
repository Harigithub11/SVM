"""
Test script for data_handler module
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_handler import (
    load_sample_dataset,
    validate_dataset,
    detect_feature_types,
    detect_target_column,
    modern_preprocessing_pipeline,
    get_dataset_info
)

def test_medical_dataset():
    print("=" * 50)
    print("Testing Medical Dataset")
    print("=" * 50)

    # Load dataset
    df = load_sample_dataset('medical')
    print(f"\n1. Dataset loaded: {df.shape}")

    # Validate
    validation = validate_dataset(df)
    print(f"\n2. Validation: {'PASS' if validation['valid'] else 'FAIL'}")
    if not validation['valid']:
        for error in validation['errors']:
            print(f"   - {error}")

    # Detect target
    target = detect_target_column(df)
    print(f"\n3. Auto-detected target column: {target}")

    # Detect features
    features = detect_feature_types(df, exclude_target=target)
    print(f"\n4. Feature types:")
    print(f"   - Numeric: {features['numeric']}")
    print(f"   - Categorical: {features['categorical']}")

    # Preprocess
    preprocessed = modern_preprocessing_pipeline(df, target_column=target)
    print(f"\n5. Preprocessing complete:")
    print(f"   - X_train shape: {preprocessed['X_train'].shape}")
    print(f"   - X_test shape: {preprocessed['X_test'].shape}")
    print(f"   - y_train shape: {preprocessed['y_train'].shape}")
    print(f"   - y_test shape: {preprocessed['y_test'].shape}")
    print(f"   - Feature names: {len(preprocessed['feature_names'])}")
    print(f"   - Class names: {preprocessed['class_names']}")

    print(f"\n[OK] Medical dataset test passed!")
    return True

def test_fraud_dataset():
    print("\n" + "=" * 50)
    print("Testing Fraud Dataset")
    print("=" * 50)

    # Load dataset
    df = load_sample_dataset('fraud')
    print(f"\n1. Dataset loaded: {df.shape}")

    # Get info
    info = get_dataset_info(df)
    print(f"\n2. Dataset info:")
    print(f"   - Memory: {info['memory_usage_mb']:.2f} MB")
    print(f"   - Duplicates: {info['duplicates']}")

    # Preprocess (with categorical features)
    preprocessed = modern_preprocessing_pipeline(df)
    print(f"\n3. Preprocessing complete:")
    print(f"   - X_train shape: {preprocessed['X_train'].shape}")
    print(f"   - X_test shape: {preprocessed['X_test'].shape}")
    print(f"   - Features: {len(preprocessed['feature_names'])}")

    print(f"\n[OK] Fraud dataset test passed!")
    return True

def test_classification_dataset():
    print("\n" + "=" * 50)
    print("Testing Classification Dataset")
    print("=" * 50)

    # Load dataset
    df = load_sample_dataset('classification')
    print(f"\n1. Dataset loaded: {df.shape}")

    # Preprocess
    preprocessed = modern_preprocessing_pipeline(df, test_size=0.25)
    print(f"\n2. Preprocessing complete (test_size=0.25):")
    print(f"   - X_train shape: {preprocessed['X_train'].shape}")
    print(f"   - X_test shape: {preprocessed['X_test'].shape}")
    print(f"   - Classes: {preprocessed['class_names']}")

    print(f"\n[OK] Classification dataset test passed!")
    return True

if __name__ == "__main__":
    print("Starting data_handler tests...\n")

    try:
        test_medical_dataset()
        test_fraud_dataset()
        test_classification_dataset()

        print("\n" + "=" * 50)
        print("[SUCCESS] All tests passed!")
        print("=" * 50)

    except Exception as e:
        print(f"\n[ERROR] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
