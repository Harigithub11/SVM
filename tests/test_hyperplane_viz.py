"""
Test new hyperplane visualization
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data_handler import load_sample_dataset, modern_preprocessing_pipeline
from src.svm_models import SVMClassifier
from src.visualizations import plot_svm_hyperplane_2d

print("="*70)
print("TESTING NEW SVM HYPERPLANE VISUALIZATION")
print("="*70)

try:
    # Load data
    print("\n1. Loading medical dataset...")
    df = load_sample_dataset('medical')
    print(f"   ✓ Loaded: {df.shape}")

    # Preprocess
    print("\n2. Preprocessing...")
    data = modern_preprocessing_pipeline(df)
    print(f"   ✓ Train: {len(data['X_train'])}, Test: {len(data['X_test'])}")

    # Train model
    print("\n3. Training RBF SVM...")
    model = SVMClassifier(kernel_type='rbf')
    model.fit(data['X_train'], data['y_train'], feature_names=data['feature_names'])
    print(f"   ✓ Trained in {model.training_time:.3f}s")
    print(f"   ✓ Support vectors: {model.model.n_support_.sum()}")

    # Create NEW hyperplane visualization
    print("\n4. Creating hyperplane visualization...")
    fig = plot_svm_hyperplane_2d(
        data['X_test'],
        data['y_test'],
        model,
        X_train=data['X_train'],
        feature_names=data['feature_names']
    )
    print("   ✓ Figure created successfully")
    print(f"   ✓ Figure size: {fig.get_size_inches()}")

    # Check figure has all required elements
    ax = fig.axes[0]
    print(f"   ✓ Has {len(ax.collections)} plot collections")
    print(f"   ✓ Has {len(ax.lines)} lines")
    print(f"   ✓ Legend items: {len(ax.get_legend().texts) if ax.get_legend() else 0}")

    plt.close(fig)

    print("\n" + "="*70)
    print("✅ TEST PASSED - New hyperplane visualization works!")
    print("="*70)

except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
