"""
Test Visualizations Module - Phase 4 Complete Testing
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from src.visualizations import (
    create_process_flowchart,
    create_interactive_flowchart,
    plot_decision_boundary_2d,
    plot_decision_boundary_3d,
    plot_confusion_matrix,
    plot_confusion_matrix_interactive,
    plot_roc_curve,
    plot_roc_curve_interactive,
    plot_metrics_bar_chart,
    plot_per_class_metrics,
    plot_train_test_comparison,
    plot_feature_importance,
    plot_support_vector_stats,
    create_results_dashboard
)

from src.data_handler import load_sample_dataset, modern_preprocessing_pipeline
from src.svm_models import (
    SVMClassifier,
    calculate_accuracy_metrics,
    calculate_confusion_matrix,
    calculate_roc_curve,
    generate_classification_report,
    generate_performance_summary
)


def test_flowcharts():
    """Test 1: Process Flowcharts"""
    print("\n" + "="*60)
    print("TEST 1: Process Flowcharts")
    print("="*60)

    # Test static flowchart
    print("\n1.1 Testing create_process_flowchart()...")
    try:
        fig = create_process_flowchart(current_step=3)
        assert fig is not None, "Figure should not be None"
        assert hasattr(fig, 'savefig'), "Should be a matplotlib figure"
        plt.close(fig)
        print("✓ Static flowchart created successfully")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test interactive flowchart
    print("\n1.2 Testing create_interactive_flowchart()...")
    try:
        fig = create_interactive_flowchart(current_step=2)
        assert fig is not None, "Figure should not be None"
        assert hasattr(fig, 'data'), "Should be a plotly figure"
        print("✓ Interactive flowchart created successfully")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    print("\n✅ Flowchart tests PASSED")
    return True


def test_decision_boundaries():
    """Test 2: Decision Boundary Plots"""
    print("\n" + "="*60)
    print("TEST 2: Decision Boundary Plots")
    print("="*60)

    # Load and prepare data
    print("\nPreparing test data...")
    try:
        df = load_sample_dataset('medical')
        data = modern_preprocessing_pipeline(df)

        # Train model
        model = SVMClassifier(kernel_type='rbf')
        model.fit(data['X_train'], data['y_train'], feature_names=data['feature_names'])
        print("✓ Model trained")
    except Exception as e:
        print(f"✗ Data preparation failed: {e}")
        return False

    # Test 2D decision boundary
    print("\n2.1 Testing plot_decision_boundary_2d()...")
    try:
        fig = plot_decision_boundary_2d(
            data['X_test'], data['y_test'], model, data['feature_names']
        )
        assert fig is not None, "Figure should not be None"
        plt.close(fig)
        print("✓ 2D decision boundary created successfully")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test 3D decision boundary
    print("\n2.2 Testing plot_decision_boundary_3d()...")
    try:
        fig = plot_decision_boundary_3d(
            data['X_test'], data['y_test'], model, data['feature_names']
        )
        assert fig is not None, "Figure should not be None"
        assert hasattr(fig, 'data'), "Should be a plotly figure"
        print("✓ 3D decision boundary created successfully")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    print("\n✅ Decision boundary tests PASSED")
    return True


def test_confusion_matrices():
    """Test 3: Confusion Matrix Visualizations"""
    print("\n" + "="*60)
    print("TEST 3: Confusion Matrix Visualizations")
    print("="*60)

    # Prepare test data
    print("\nPreparing test data...")
    try:
        df = load_sample_dataset('classification')
        data = modern_preprocessing_pipeline(df)
        model = SVMClassifier(kernel_type='linear')
        model.fit(data['X_train'], data['y_train'])

        y_pred = model.predict(data['X_test'])
        cm = calculate_confusion_matrix(data['y_test'], y_pred)
        print("✓ Test data prepared")
    except Exception as e:
        print(f"✗ Data preparation failed: {e}")
        return False

    # Test static confusion matrix
    print("\n3.1 Testing plot_confusion_matrix()...")
    try:
        fig = plot_confusion_matrix(
            data['y_test'], y_pred,
            class_names=data['class_names']
        )
        assert fig is not None, "Figure should not be None"
        plt.close(fig)
        print("✓ Static confusion matrix created successfully")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test normalized confusion matrix
    print("\n3.2 Testing plot_confusion_matrix() with normalization...")
    try:
        fig = plot_confusion_matrix(
            data['y_test'], y_pred,
            class_names=data['class_names'],
            normalize='true'
        )
        assert fig is not None, "Figure should not be None"
        plt.close(fig)
        print("✓ Normalized confusion matrix created successfully")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test interactive confusion matrix
    print("\n3.3 Testing plot_confusion_matrix_interactive()...")
    try:
        fig = plot_confusion_matrix_interactive(cm, class_names=data['class_names'])
        assert fig is not None, "Figure should not be None"
        assert hasattr(fig, 'data'), "Should be a plotly figure"
        print("✓ Interactive confusion matrix created successfully")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    print("\n✅ Confusion matrix tests PASSED")
    return True


def test_roc_curves():
    """Test 4: ROC Curve Visualizations"""
    print("\n" + "="*60)
    print("TEST 4: ROC Curve Visualizations")
    print("="*60)

    # Prepare test data
    print("\nPreparing test data...")
    try:
        df = load_sample_dataset('fraud')
        data = modern_preprocessing_pipeline(df)
        model = SVMClassifier(kernel_type='rbf')
        model.fit(data['X_train'], data['y_train'])

        y_pred_proba = model.predict_proba(data['X_test'])
        roc_data = calculate_roc_curve(data['y_test'], y_pred_proba)
        print("✓ Test data prepared")
    except Exception as e:
        print(f"✗ Data preparation failed: {e}")
        return False

    # Test static ROC curve
    print("\n4.1 Testing plot_roc_curve()...")
    try:
        fig = plot_roc_curve(roc_data)
        assert fig is not None, "Figure should not be None"
        plt.close(fig)
        print("✓ Static ROC curve created successfully")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test interactive ROC curve
    print("\n4.2 Testing plot_roc_curve_interactive()...")
    try:
        fig = plot_roc_curve_interactive(roc_data)
        assert fig is not None, "Figure should not be None"
        assert hasattr(fig, 'data'), "Should be a plotly figure"
        print("✓ Interactive ROC curve created successfully")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    print("\n✅ ROC curve tests PASSED")
    return True


def test_performance_metrics():
    """Test 5: Performance Metrics Visualizations"""
    print("\n" + "="*60)
    print("TEST 5: Performance Metrics Visualizations")
    print("="*60)

    # Prepare test data
    print("\nPreparing test data...")
    try:
        df = load_sample_dataset('medical')
        data = modern_preprocessing_pipeline(df)
        model = SVMClassifier(kernel_type='poly')
        model.fit(data['X_train'], data['y_train'])

        summary = generate_performance_summary(
            model,
            data['X_train'], data['y_train'],
            data['X_test'], data['y_test']
        )
        print("✓ Test data prepared")
    except Exception as e:
        print(f"✗ Data preparation failed: {e}")
        return False

    # Test metrics bar chart
    print("\n5.1 Testing plot_metrics_bar_chart()...")
    try:
        fig = plot_metrics_bar_chart(summary['test_metrics'])
        assert fig is not None, "Figure should not be None"
        plt.close(fig)
        print("✓ Metrics bar chart created successfully")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test per-class metrics
    print("\n5.2 Testing plot_per_class_metrics()...")
    try:
        fig = plot_per_class_metrics(summary['classification_report'])
        assert fig is not None, "Figure should not be None"
        plt.close(fig)
        print("✓ Per-class metrics chart created successfully")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test train-test comparison
    print("\n5.3 Testing plot_train_test_comparison()...")
    try:
        fig = plot_train_test_comparison(
            summary['train_metrics'],
            summary['test_metrics']
        )
        assert fig is not None, "Figure should not be None"
        plt.close(fig)
        print("✓ Train-test comparison chart created successfully")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    print("\n✅ Performance metrics tests PASSED")
    return True


def test_feature_importance():
    """Test 6: Feature Importance Visualization"""
    print("\n" + "="*60)
    print("TEST 6: Feature Importance Visualization")
    print("="*60)

    # Prepare test data with linear kernel
    print("\nPreparing test data (linear kernel)...")
    try:
        df = load_sample_dataset('classification')
        data = modern_preprocessing_pipeline(df)
        model = SVMClassifier(kernel_type='linear')
        model.fit(data['X_train'], data['y_train'], feature_names=data['feature_names'])
        print("✓ Test data prepared")
    except Exception as e:
        print(f"✗ Data preparation failed: {e}")
        return False

    # Test feature importance
    print("\n6.1 Testing plot_feature_importance()...")
    try:
        fig = plot_feature_importance(model, data['feature_names'], top_n=10)
        assert fig is not None, "Figure should not be None"
        plt.close(fig)
        print("✓ Feature importance plot created successfully")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test with non-linear kernel (should handle gracefully)
    print("\n6.2 Testing plot_feature_importance() with non-linear kernel...")
    try:
        model_rbf = SVMClassifier(kernel_type='rbf')
        model_rbf.fit(data['X_train'], data['y_train'], feature_names=data['feature_names'])
        fig = plot_feature_importance(model_rbf, data['feature_names'])
        assert fig is not None, "Should return a figure with message"
        plt.close(fig)
        print("✓ Non-linear kernel handled gracefully")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    print("\n✅ Feature importance tests PASSED")
    return True


def test_support_vector_stats():
    """Test 7: Support Vector Statistics"""
    print("\n" + "="*60)
    print("TEST 7: Support Vector Statistics")
    print("="*60)

    # Prepare test data
    print("\nPreparing test data...")
    try:
        df = load_sample_dataset('medical')
        data = modern_preprocessing_pipeline(df)
        model = SVMClassifier(kernel_type='rbf')
        model.fit(data['X_train'], data['y_train'])
        print("✓ Test data prepared")
    except Exception as e:
        print(f"✗ Data preparation failed: {e}")
        return False

    # Test support vector stats
    print("\n7.1 Testing plot_support_vector_stats()...")
    try:
        fig = plot_support_vector_stats(model, data['y_train'])
        assert fig is not None, "Figure should not be None"
        plt.close(fig)
        print("✓ Support vector stats plot created successfully")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    print("\n✅ Support vector stats tests PASSED")
    return True


def test_results_dashboard():
    """Test 8: Complete Results Dashboard"""
    print("\n" + "="*60)
    print("TEST 8: Complete Results Dashboard")
    print("="*60)

    # Prepare comprehensive test data
    print("\nPreparing comprehensive test data...")
    try:
        df = load_sample_dataset('classification')
        data = modern_preprocessing_pipeline(df)
        model = SVMClassifier(kernel_type='linear')
        model.fit(data['X_train'], data['y_train'], feature_names=data['feature_names'])

        summary = generate_performance_summary(
            model,
            data['X_train'], data['y_train'],
            data['X_test'], data['y_test']
        )
        print("✓ Test data prepared")
    except Exception as e:
        print(f"✗ Data preparation failed: {e}")
        return False

    # Test dashboard creation
    print("\n8.1 Testing create_results_dashboard()...")
    try:
        figures = create_results_dashboard(
            model,
            data['X_train'], data['y_train'],
            data['X_test'], data['y_test'],
            summary['test_metrics'],
            summary['confusion_matrix'],
            summary['roc_curve_data'],
            feature_names=data['feature_names'],
            class_names=data['class_names']
        )

        assert isinstance(figures, dict), "Should return a dictionary"
        assert len(figures) > 0, "Should contain at least one figure"

        print(f"✓ Dashboard created with {len(figures)} visualizations")

        # List all figures created
        print("\n   Visualizations included:")
        for key in figures.keys():
            print(f"   - {key}")

        # Close all figures
        for fig in figures.values():
            if hasattr(fig, 'savefig'):  # matplotlib figure
                plt.close(fig)

        print("\n✓ Results dashboard created successfully")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    print("\n✅ Results dashboard tests PASSED")
    return True


def run_all_tests():
    """Run all Phase 4 visualization tests"""
    print("\n" + "="*70)
    print("PHASE 4 VISUALIZATION MODULE - COMPLETE TEST SUITE")
    print("="*70)

    results = {}

    # Run all tests
    results['Flowcharts'] = test_flowcharts()
    results['Decision Boundaries'] = test_decision_boundaries()
    results['Confusion Matrices'] = test_confusion_matrices()
    results['ROC Curves'] = test_roc_curves()
    results['Performance Metrics'] = test_performance_metrics()
    results['Feature Importance'] = test_feature_importance()
    results['Support Vector Stats'] = test_support_vector_stats()
    results['Results Dashboard'] = test_results_dashboard()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<50} {status}")

    print("\n" + "-"*70)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print("="*70)

    if failed == 0:
        print("\n🎉 ALL PHASE 4 TESTS PASSED! Phase 4 is COMPLETE!")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Phase 4 is NOT complete.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
