"""
Test script for svm_models module
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_handler import load_sample_dataset, modern_preprocessing_pipeline
from svm_models import (
    SVMClassifier,
    train_svm_model,
    optimize_svm_halving,
    generate_performance_summary
)

def test_basic_svm():
    print("=" * 50)
    print("Test 1: Basic SVM Training (All Kernels)")
    print("=" * 50)

    # Load and preprocess data
    df = load_sample_dataset('classification')
    data = modern_preprocessing_pipeline(df)

    kernels = ['linear', 'rbf', 'poly']

    for kernel in kernels:
        print(f"\n--- Testing {kernel.upper()} kernel ---")

        # Train model
        model = train_svm_model(
            data['X_train'],
            data['y_train'],
            kernel_type=kernel,
            feature_names=data['feature_names']
        )

        print(f"Training time: {model.training_time:.3f} seconds")

        # Predictions
        y_pred = model.predict(data['X_test'])
        y_proba = model.predict_proba(data['X_test'])

        print(f"Predictions shape: {y_pred.shape}")
        print(f"Probabilities shape: {y_proba.shape}")

        # Support vectors
        sv_info = model.get_support_vectors()
        if sv_info:
            print(f"Support vectors per class: {sv_info['n_support']}")

        print(f"[OK] {kernel.upper()} kernel test passed")

    return True

def test_performance_summary():
    print("\n" + "=" * 50)
    print("Test 2: Performance Summary")
    print("=" * 50)

    # Load medical dataset
    df = load_sample_dataset('medical')
    data = modern_preprocessing_pipeline(df)

    # Train RBF model
    model = train_svm_model(
        data['X_train'],
        data['y_train'],
        kernel_type='rbf'
    )

    # Generate summary
    summary = generate_performance_summary(
        model,
        data['X_train'],
        data['y_train'],
        data['X_test'],
        data['y_test']
    )

    print(f"\nTrain Accuracy: {summary['train_metrics']['accuracy']:.3f}")
    print(f"Test Accuracy: {summary['test_metrics']['accuracy']:.3f}")
    print(f"Test Precision: {summary['test_metrics']['precision']:.3f}")
    print(f"Test Recall: {summary['test_metrics']['recall']:.3f}")
    print(f"Test F1-Score: {summary['test_metrics']['f1_score']:.3f}")

    print(f"\nConfusion Matrix shape: {summary['confusion_matrix'].shape}")

    # Check ROC data
    roc_data = summary['roc_curve_data']
    if 'binary' in roc_data:
        print(f"Binary ROC AUC: {roc_data['binary']['auc']:.3f}")
    elif 'summary' in roc_data:
        print(f"Micro-average AUC: {roc_data['summary']['micro_auc']:.3f}")
        print(f"Macro-average AUC: {roc_data['summary']['macro_auc']:.3f}")

    print(f"\n[OK] Performance summary test passed")
    return True

def test_halving_grid_search():
    print("\n" + "=" * 50)
    print("Test 3: HalvingGridSearchCV Optimization")
    print("=" * 50)

    # Load classification dataset
    df = load_sample_dataset('classification')
    data = modern_preprocessing_pipeline(df)

    # Optimize with HalvingGridSearchCV
    print("\nRunning HalvingGridSearchCV (RBF kernel)...")
    results = optimize_svm_halving(
        data['X_train'],
        data['y_train'],
        kernel='rbf',
        cv=3
    )

    print(f"\nBest parameters: {results['best_params']}")
    print(f"Best CV score: {results['best_score']:.3f}")
    print(f"Search time: {results['search_time']:.3f} seconds")

    # Test best model
    best_model = results['best_estimator']
    test_score = best_model.score(data['X_test'], data['y_test'])
    print(f"Test score with best params: {test_score:.3f}")

    print(f"\n[OK] HalvingGridSearchCV test passed")
    return True

def test_fraud_dataset():
    print("\n" + "=" * 50)
    print("Test 4: Fraud Dataset (Categorical Features)")
    print("=" * 50)

    # Load fraud dataset (has categorical 'Location' column)
    df = load_sample_dataset('fraud')
    data = modern_preprocessing_pipeline(df)

    print(f"\nPreprocessed features: {data['X_train'].shape[1]}")
    print(f"Original features: {len(data['feature_names'])}")

    # Train model
    model = train_svm_model(
        data['X_train'],
        data['y_train'],
        kernel_type='rbf'
    )

    # Generate summary
    summary = generate_performance_summary(
        model,
        data['X_train'],
        data['y_train'],
        data['X_test'],
        data['y_test']
    )

    print(f"\nTest Accuracy: {summary['test_metrics']['accuracy']:.3f}")
    print(f"Number of classes: {len(data['class_names'])}")

    # Check multi-class ROC
    if 'summary' in summary['roc_curve_data']:
        print(f"Per-class AUC: {summary['roc_curve_data']['summary']['per_class_auc']}")

    print(f"\n[OK] Fraud dataset test passed")
    return True

if __name__ == "__main__":
    print("Starting svm_models tests...\n")

    try:
        test_basic_svm()
        test_performance_summary()
        test_halving_grid_search()
        test_fraud_dataset()

        print("\n" + "=" * 50)
        print("[SUCCESS] All SVM tests passed!")
        print("=" * 50)

    except Exception as e:
        print(f"\n[ERROR] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
