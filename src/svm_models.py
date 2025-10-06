"""
SVM Models Module
Implements SVM classification with multiple kernels and modern optimizations
"""

import numpy as np
import pandas as pd
import time
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import warnings
warnings.filterwarnings('ignore')

class SVMClassifier:
    """
    SVM Classifier with multiple kernel support
    """

    def __init__(self, kernel_type='rbf', C=1.0, gamma='scale', degree=3, random_state=42):
        """
        Initialize SVM Classifier

        Args:
            kernel_type (str): Kernel type ('linear', 'rbf', 'poly')
            C (float): Regularization parameter
            gamma (str/float): Kernel coefficient
            degree (int): Degree for polynomial kernel
            random_state (int): Random seed
        """
        self.kernel_type = kernel_type
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.random_state = random_state
        self.model = None
        self.training_time = 0
        self.feature_names = None

    def fit(self, X_train, y_train, feature_names=None):
        """
        Train the SVM model

        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: List of feature names
        """
        self.feature_names = feature_names

        start_time = time.time()

        # Use optimal SVM selection based on data size and kernel
        n_samples = X_train.shape[0]
        self.model = create_optimal_svm(
            kernel=self.kernel_type,
            n_samples=n_samples,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            random_state=self.random_state
        )

        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time

        return self

    def predict(self, X):
        """Predict class labels"""
        if self.model is None:
            raise Exception("Model not trained yet!")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.model is None:
            raise Exception("Model not trained yet!")

        # Check if model supports probability
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # For LinearSVC and SGDClassifier, use decision function
            decision = self.model.decision_function(X)
            # Convert to probabilities using softmax
            if len(decision.shape) == 1:
                decision = decision.reshape(-1, 1)
                decision = np.hstack([-decision, decision])
            exp_dec = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp_dec / exp_dec.sum(axis=1, keepdims=True)
        else:
            raise Exception("Model does not support probability prediction")

    def decision_function(self, X):
        """Get decision function values"""
        if self.model is None:
            raise Exception("Model not trained yet!")
        return self.model.decision_function(X)

    @property
    def classes_(self):
        """Get class labels"""
        if self.model is None:
            raise Exception("Model not trained yet!")
        return self.model.classes_

    def get_support_vectors(self):
        """Get support vectors (if available)"""
        if hasattr(self.model, 'support_vectors_'):
            return {
                'support_vectors': self.model.support_vectors_,
                'support_indices': self.model.support_,
                'n_support': self.model.n_support_
            }
        return None


def create_optimal_svm(kernel='rbf', n_samples=None, **kwargs):
    """
    Create SVM with optimal algorithm selection (2024)

    Args:
        kernel (str): Kernel type
        n_samples (int): Number of samples
        **kwargs: Additional parameters

    Returns:
        Trained model
    """
    # Large dataset optimization (>100k samples)
    if n_samples and n_samples > 100000:
        return SGDClassifier(
            loss='hinge',
            max_iter=1000,
            random_state=kwargs.get('random_state', 42),
            **{k: v for k, v in kwargs.items() if k != 'random_state'}
        )

    # LinearSVC for linear kernel with moderate/large data (>10k samples)
    elif kernel == 'linear' and n_samples and n_samples > 10000:
        return LinearSVC(
            dual=False,
            max_iter=1000,
            random_state=kwargs.get('random_state', 42),
            C=kwargs.get('C', 1.0)
        )

    # Standard SVC for small/medium datasets or non-linear kernels
    else:
        return SVC(
            kernel=kernel,
            C=kwargs.get('C', 1.0),
            gamma=kwargs.get('gamma', 'scale'),
            degree=kwargs.get('degree', 3),
            cache_size=500,
            random_state=kwargs.get('random_state', 42),
            probability=True  # Enable probability estimates
        )


def train_svm_model(X_train, y_train, kernel_type='rbf', feature_names=None, **params):
    """
    Train SVM model

    Args:
        X_train: Training features
        y_train: Training labels
        kernel_type (str): Kernel type
        feature_names: Feature names
        **params: Additional parameters

    Returns:
        Trained SVMClassifier
    """
    model = SVMClassifier(kernel_type=kernel_type, **params)
    model.fit(X_train, y_train, feature_names=feature_names)
    return model


def optimize_svm_halving(X_train, y_train, kernel='rbf', cv=3):
    """
    Optimize SVM using HalvingGridSearchCV (3-5x faster)

    Args:
        X_train: Training features
        y_train: Training labels
        kernel (str): Kernel type
        cv (int): Cross-validation folds

    Returns:
        dict: Optimization results
    """
    # Define parameter grids
    param_grids = {
        'linear': {
            'C': np.logspace(-2, 2, 10),
            'class_weight': [None, 'balanced']
        },
        'rbf': {
            'C': np.logspace(-2, 2, 10),
            'gamma': np.logspace(-3, 1, 10),
            'class_weight': [None, 'balanced']
        },
        'poly': {
            'C': np.logspace(-2, 2, 8),
            'gamma': np.logspace(-3, 1, 8),
            'degree': [2, 3, 4],
            'class_weight': [None, 'balanced']
        }
    }

    # Create base model
    base_model = SVC(kernel=kernel, cache_size=500, random_state=42, probability=True)

    # HalvingGridSearchCV
    search = HalvingGridSearchCV(
        base_model,
        param_grids.get(kernel, param_grids['rbf']),
        cv=cv,
        factor=3,
        resource='n_samples',
        max_resources='auto',
        random_state=42,
        n_jobs=-1
    )

    start_time = time.time()
    search.fit(X_train, y_train)
    search_time = time.time() - start_time

    return {
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'best_estimator': search.best_estimator_,
        'search_time': search_time,
        'cv_results': search.cv_results_
    }


def calculate_accuracy_metrics(y_true, y_pred):
    """
    Calculate basic accuracy metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        dict: Accuracy metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    return metrics


def generate_classification_report(y_true, y_pred, target_names=None):
    """
    Generate detailed classification report

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Class names

    Returns:
        dict: Classification report as dictionary
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    return report


def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        array: Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm


def calculate_roc_curve(y_true, y_pred_proba):
    """
    Calculate ROC curve data with micro/macro averaging for multi-class (2024)

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities

    Returns:
        dict: ROC curve data for each class + micro/macro averages
    """
    n_classes = len(np.unique(y_true))
    roc_data = {}

    if n_classes == 2:
        # Binary classification
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
        roc_data['binary'] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc_score(y_true, y_pred_proba[:, 1])
        }
    else:
        # Multi-class: One-vs-Rest with micro/macro averaging
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)

        # Per-class ROC curves
        fpr_dict = {}
        tpr_dict = {}
        roc_auc_dict = {}

        for i in range(n_classes):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

            roc_data[f'class_{i}'] = {
                'fpr': fpr_dict[i],
                'tpr': tpr_dict[i],
                'auc': roc_auc_dict[i],
                'class_label': classes[i]
            }

        # Micro-average ROC curve (aggregate all classes)
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)

        roc_data['micro'] = {
            'fpr': fpr_micro,
            'tpr': tpr_micro,
            'auc': roc_auc_micro,
            'description': 'Micro-average (all classes aggregated)'
        }

        # Macro-average ROC curve (average of per-class ROCs)
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= n_classes
        roc_auc_macro = auc(all_fpr, mean_tpr)

        roc_data['macro'] = {
            'fpr': all_fpr,
            'tpr': mean_tpr,
            'auc': roc_auc_macro,
            'description': 'Macro-average (mean of per-class ROCs)'
        }

        # Summary statistics
        roc_data['summary'] = {
            'n_classes': n_classes,
            'micro_auc': roc_auc_micro,
            'macro_auc': roc_auc_macro,
            'per_class_auc': roc_auc_dict
        }

    return roc_data


def generate_performance_summary(model, X_train, y_train, X_test, y_test):
    """
    Generate comprehensive performance summary

    Args:
        model: Trained SVMClassifier
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels

    Returns:
        dict: Performance summary
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Probabilities
    y_train_proba = model.predict_proba(X_train)
    y_test_proba = model.predict_proba(X_test)

    # Metrics
    train_metrics = calculate_accuracy_metrics(y_train, y_train_pred)
    test_metrics = calculate_accuracy_metrics(y_test, y_test_pred)

    # Confusion matrix
    cm = calculate_confusion_matrix(y_test, y_test_pred)

    # ROC curve
    roc_data = calculate_roc_curve(y_test, y_test_proba)

    # Classification report
    report = generate_classification_report(y_test, y_test_pred)

    summary = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'confusion_matrix': cm,
        'roc_curve_data': roc_data,
        'classification_report': report,
        'training_time': model.training_time,
        'n_support_vectors': model.get_support_vectors()['n_support'] if model.get_support_vectors() else None
    }

    return summary
