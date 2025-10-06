# Phase 4: Visualizations

## Overview
Create comprehensive visualization functions for process flowcharts, decision boundaries, confusion matrices, ROC curves, performance metrics, feature importance, and support vectors.

**Estimated Duration:** 5-6 hours

**Prerequisites:** Phase 1, 2, & 3 completed

---

## 1. Imports and Dependencies

### Required Imports for Visualization Module

```python
# File: src/visualizations.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay  # NEW: sklearn 1.7.2 built-in
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
```

---

## 2. Process Flowchart

### 2.1 Static Flowchart

#### Function: `create_process_flowchart()`
```python
def create_process_flowchart(current_step=None):
    """
    Create process flowchart showing workflow steps

    Args:
        current_step (int): Current step to highlight (1-5)

    Returns:
        matplotlib.figure.Figure: Flowchart figure
    """
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis('off')

    steps = [
        '1. Select\nDomain',
        '2. Choose\nKernel',
        '3. Upload\nDataset',
        '4. Process\n& Train',
        '5. View\nResults'
    ]

    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    positions = np.linspace(0.1, 0.9, len(steps))

    for i, (step, pos, color) in enumerate(zip(steps, positions, colors)):
        # Determine if this step should be highlighted
        alpha = 1.0 if current_step is None or current_step == i + 1 else 0.3

        # Draw box
        box = plt.Rectangle((pos - 0.08, 0.3), 0.16, 0.4,
                            facecolor=color, edgecolor='black',
                            linewidth=2, alpha=alpha)
        ax.add_patch(box)

        # Add text
        ax.text(pos, 0.5, step, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

        # Add arrow to next step
        if i < len(steps) - 1:
            ax.annotate('', xy=(positions[i + 1] - 0.08, 0.5),
                       xytext=(pos + 0.08, 0.5),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black', alpha=alpha))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    return fig
```

---

### 2.2 Interactive Flowchart (Plotly)

#### Function: `create_interactive_flowchart()`
```python
def create_interactive_flowchart(current_step=None):
    """
    Create interactive flowchart using Plotly

    Args:
        current_step (int): Current step to highlight (1-5)

    Returns:
        plotly.graph_objects.Figure: Interactive flowchart
    """
    steps = ['Domain Selection', 'Kernel Selection', 'Upload Dataset',
             'Process & Train', 'View Results']

    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

    # Create figure
    fig = go.Figure()

    # Add boxes for each step
    for i, (step, color) in enumerate(zip(steps, colors)):
        opacity = 1.0 if current_step is None or current_step == i + 1 else 0.3

        fig.add_trace(go.Scatter(
            x=[i], y=[0],
            mode='markers+text',
            marker=dict(size=80, color=color, opacity=opacity,
                       line=dict(color='black', width=2)),
            text=f'{i+1}. {step}',
            textposition='middle center',
            textfont=dict(size=10, color='white', family='Arial Black'),
            showlegend=False,
            hovertext=step,
            hoverinfo='text'
        ))

        # Add arrow
        if i < len(steps) - 1:
            fig.add_annotation(
                x=i + 0.5, y=0,
                ax=i, ay=0,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor='black',
                opacity=opacity
            )

    fig.update_layout(
        title='SVM Application Workflow',
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white'
    )

    return fig
```

---

## 3. Decision Boundary Plots

### 3.1 2D Decision Boundary

#### Function: `plot_decision_boundary_2d()`
```python
def plot_decision_boundary_2d(model, X, y, feature_names=None, resolution=0.02):
    """
    Plot 2D decision boundary

    Args:
        model: Trained SVM model
        X: Feature data (will use first 2 features or PCA)
        y: Labels
        feature_names: Feature names
        resolution: Mesh resolution

    Returns:
        matplotlib.figure.Figure: Decision boundary plot
    """
    # Reduce to 2D if needed
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.2%})'
        ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'
    else:
        X_2d = X
        xlabel = feature_names[0] if feature_names else 'Feature 1'
        ylabel = feature_names[1] if feature_names and len(feature_names) > 1 else 'Feature 2'

    # Create mesh
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))

    # Predict on mesh
    if X.shape[1] > 2:
        # Inverse transform for prediction
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(pca.inverse_transform(mesh_points))
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Decision boundary
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    ax.contour(xx, yy, Z, colors='black', linewidths=0.5, alpha=0.5)

    # Scatter plot
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.RdYlBu,
                        edgecolors='black', s=50, alpha=0.8)

    # Highlight support vectors
    if hasattr(model, 'model') and hasattr(model.model, 'support_'):
        support_vectors_2d = X_2d[model.model.support_]
        ax.scatter(support_vectors_2d[:, 0], support_vectors_2d[:, 1],
                  s=200, linewidth=2, facecolors='none',
                  edgecolors='green', label='Support Vectors')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title('SVM Decision Boundary (2D)', fontsize=14, fontweight='bold')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Class')
    plt.tight_layout()

    return fig
```

---

### 3.2 3D Decision Boundary

#### Function: `plot_decision_boundary_3d()`
```python
def plot_decision_boundary_3d(model, X, y, feature_names=None):
    """
    Plot 3D decision boundary using Plotly

    Args:
        model: Trained SVM model
        X: Feature data (will use first 3 features or PCA)
        y: Labels
        feature_names: Feature names

    Returns:
        plotly.graph_objects.Figure: 3D decision boundary plot
    """
    # Reduce to 3D if needed
    if X.shape[1] > 3:
        pca = PCA(n_components=3)
        X_3d = pca.fit_transform(X)
        labels = [f'PC{i+1} ({pca.explained_variance_ratio_[i]:.2%})'
                 for i in range(3)]
    else:
        X_3d = X[:, :3]
        if feature_names and len(feature_names) >= 3:
            labels = feature_names[:3]
        else:
            labels = [f'Feature {i+1}' for i in range(3)]

    # Create scatter plot
    fig = go.Figure()

    # Plot each class
    for class_val in np.unique(y):
        mask = y == class_val
        fig.add_trace(go.Scatter3d(
            x=X_3d[mask, 0],
            y=X_3d[mask, 1],
            z=X_3d[mask, 2],
            mode='markers',
            name=f'Class {class_val}',
            marker=dict(size=5, opacity=0.8)
        ))

    # Highlight support vectors
    if hasattr(model, 'model') and hasattr(model.model, 'support_'):
        sv_indices = model.model.support_
        fig.add_trace(go.Scatter3d(
            x=X_3d[sv_indices, 0],
            y=X_3d[sv_indices, 1],
            z=X_3d[sv_indices, 2],
            mode='markers',
            name='Support Vectors',
            marker=dict(
                size=8,
                color='green',
                symbol='diamond',
                line=dict(color='black', width=2)
            )
        ))

    fig.update_layout(
        title='SVM Decision Boundary (3D)',
        scene=dict(
            xaxis_title=labels[0],
            yaxis_title=labels[1],
            zaxis_title=labels[2]
        ),
        height=600,
        showlegend=True
    )

    return fig
```

---

## 4. Confusion Matrix Visualization

### 4.1 Confusion Matrix Heatmap (Modern - sklearn 1.7.2)

#### Function: `plot_confusion_matrix()` - UPDATED with ConfusionMatrixDisplay
```python
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=None,
                         title='Confusion Matrix', cmap='Blues'):
    """
    Plot confusion matrix using sklearn's ConfusionMatrixDisplay (2024 best practice)

    Enhancements from sklearn 1.7.2:
    - Built-in normalization options: None, 'true', 'pred', 'all'
    - Automatic text formatting based on value ranges
    - Better colorbar handling
    - Consistent API with sklearn ecosystem

    Args:
        y_true: True labels (NOT confusion matrix - changed API!)
        y_pred: Predicted labels
        class_names: Display names for classes
        normalize: None, 'true', 'pred', or 'all'
        title: Plot title
        cmap: Colormap (default: 'Blues')

    Returns:
        matplotlib.figure.Figure: Confusion matrix figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # **NEW: Use ConfusionMatrixDisplay from sklearn 1.7.2**
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        normalize=normalize,  # None, 'true', 'pred', or 'all'
        cmap=cmap,
        ax=ax,
        colorbar=True,
        values_format=None  # Auto-format: '.2g' for normalized, 'd' for counts
    )

    # Customize labels
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add grid for better readability
    ax.grid(False)

    plt.tight_layout()
    return fig

# **Alternative: From pre-computed confusion matrix**
def plot_confusion_matrix_from_cm(cm, class_names=None, normalize=None,
                                  title='Confusion Matrix', cmap='Blues'):
    """
    Plot from pre-computed confusion matrix

    Args:
        cm: Pre-computed confusion matrix
        class_names: Display names
        normalize: None, 'true', 'pred', or 'all'
        title: Plot title
        cmap: Colormap

    Returns:
        matplotlib.figure.Figure: Confusion matrix figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalize if requested
    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        cm = cm.astype('float') / cm.sum()

    # Use ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    disp.plot(cmap=cmap, ax=ax, colorbar=True)

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig
```

#### Why Use ConfusionMatrixDisplay?

**Benefits over manual seaborn/matplotlib:**
1. **Built-in normalization**: `normalize='true'` (per-row), `'pred'` (per-column), `'all'` (global)
2. **Automatic formatting**: Smart value display based on normalized vs raw
3. **Consistent API**: Matches sklearn ecosystem patterns
4. **Less code**: No manual formatting logic needed
5. **Better defaults**: Optimized colors and labels

**Usage Example:**
```python
# Modern approach (recommended)
fig = plot_confusion_matrix(y_test, y_pred, class_names=['Class 0', 'Class 1', 'Class 2'],
                           normalize='true')  # Normalize by true labels (rows)

# Or from pre-computed matrix
cm = confusion_matrix(y_test, y_pred)
fig = plot_confusion_matrix_from_cm(cm, class_names=['Class 0', 'Class 1', 'Class 2'])
```

---

### 4.2 Interactive Confusion Matrix

#### Function: `plot_confusion_matrix_interactive()`
```python
def plot_confusion_matrix_interactive(cm, class_names=None, normalize=False):
    """
    Create interactive confusion matrix using Plotly

    Args:
        cm: Confusion matrix
        class_names: Class labels
        normalize: Whether to normalize

    Returns:
        plotly.graph_objects.Figure: Interactive confusion matrix
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]

    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        text_template = '%{z:.2%}'
    else:
        cm_display = cm
        text_template = '%{z}'

    fig = go.Figure(data=go.Heatmap(
        z=cm_display,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm_display,
        texttemplate=text_template,
        textfont={"size": 14},
        colorbar=dict(title='Percentage' if normalize else 'Count')
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=500,
        width=600
    )

    return fig
```

---

## 5. ROC Curve Visualization

### 5.1 ROC Curve Plot

#### Function: `plot_roc_curve()`
```python
def plot_roc_curve(roc_data, title='ROC Curve'):
    """
    Plot ROC curve

    Args:
        roc_data: ROC data dictionary from calculate_roc_curve()
        title: Plot title

    Returns:
        matplotlib.figure.Figure: ROC curve plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)

    # Plot ROC curves
    if 'binary' in roc_data:
        # Binary classification
        data = roc_data['binary']
        ax.plot(data['fpr'], data['tpr'],
               label=f"ROC (AUC = {data['auc']:.3f})",
               linewidth=2)
    else:
        # Multi-class
        colors = plt.cm.rainbow(np.linspace(0, 1, len(roc_data)))
        for (class_name, data), color in zip(roc_data.items(), colors):
            ax.plot(data['fpr'], data['tpr'],
                   label=f"{class_name} (AUC = {data['auc']:.3f})",
                   color=color, linewidth=2)

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig
```

---

### 5.2 Interactive ROC Curve

#### Function: `plot_roc_curve_interactive()`
```python
def plot_roc_curve_interactive(roc_data):
    """
    Create interactive ROC curve using Plotly

    Args:
        roc_data: ROC data dictionary

    Returns:
        plotly.graph_objects.Figure: Interactive ROC curve
    """
    fig = go.Figure()

    # Add diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))

    # Add ROC curves
    if 'binary' in roc_data:
        data = roc_data['binary']
        fig.add_trace(go.Scatter(
            x=data['fpr'],
            y=data['tpr'],
            mode='lines',
            name=f"ROC (AUC = {data['auc']:.3f})",
            line=dict(width=2),
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))
    else:
        for class_name, data in roc_data.items():
            fig.add_trace(go.Scatter(
                x=data['fpr'],
                y=data['tpr'],
                mode='lines',
                name=f"{class_name} (AUC = {data['auc']:.3f})",
                line=dict(width=2),
                hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
            ))

    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=600,
        width=700,
        hovermode='closest'
    )

    return fig
```

---

## 6. Performance Metrics Visualization

### 6.1 Metrics Bar Chart

#### Function: `plot_metrics_bar_chart()`
```python
def plot_metrics_bar_chart(metrics_dict):
    """
    Plot performance metrics as bar chart

    Args:
        metrics_dict: Dictionary with metrics (accuracy, precision, recall, f1)

    Returns:
        matplotlib.figure.Figure: Metrics bar chart
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        metrics_dict.get('accuracy', 0),
        metrics_dict.get('precision', 0),
        metrics_dict.get('recall', 0),
        metrics_dict.get('f1_score', 0)
    ]

    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    return fig
```

---

### 6.2 Per-Class Metrics Chart

#### Function: `plot_per_class_metrics()`
```python
def plot_per_class_metrics(classification_report):
    """
    Plot per-class metrics

    Args:
        classification_report: Classification report dictionary

    Returns:
        matplotlib.figure.Figure: Per-class metrics chart
    """
    # Extract per-class metrics
    classes = [key for key in classification_report.keys()
              if key not in ['accuracy', 'macro avg', 'weighted avg']]

    precision = [classification_report[cls]['precision'] for cls in classes]
    recall = [classification_report[cls]['recall'] for cls in classes]
    f1 = [classification_report[cls]['f1-score'] for cls in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', color='#2ecc71', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', color='#f39c12', alpha=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    return fig
```

---

### 6.3 Training vs Testing Comparison

#### Function: `plot_train_test_comparison()`
```python
def plot_train_test_comparison(train_metrics, test_metrics):
    """
    Compare training and testing metrics

    Args:
        train_metrics: Training metrics dictionary
        test_metrics: Testing metrics dictionary

    Returns:
        matplotlib.figure.Figure: Comparison chart
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    train_values = [
        train_metrics.get('accuracy', 0),
        train_metrics.get('precision', 0),
        train_metrics.get('recall', 0),
        train_metrics.get('f1_score', 0)
    ]
    test_values = [
        test_metrics.get('accuracy', 0),
        test_metrics.get('precision', 0),
        test_metrics.get('recall', 0),
        test_metrics.get('f1_score', 0)
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width/2, train_values, width, label='Training', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, test_values, width, label='Testing', color='#e74c3c', alpha=0.8)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Training vs Testing Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()

    return fig
```

---

## 7. Feature Importance Visualization

### 7.1 Feature Importance Bar Plot

#### Function: `plot_feature_importance()`
```python
def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot feature importance for linear SVM

    Args:
        model: Trained SVM model
        feature_names: List of feature names
        top_n: Number of top features to show

    Returns:
        matplotlib.figure.Figure: Feature importance plot
    """
    if not hasattr(model.model, 'coef_'):
        # Non-linear kernel, use approximation
        return None

    # Get feature importance (absolute coefficients)
    importance = np.abs(model.model.coef_[0])

    # Create DataFrame
    feature_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_df)))
    bars = ax.barh(feature_df['feature'], feature_df['importance'],
                   color=colors, alpha=0.8, edgecolor='black')

    ax.set_xlabel('Importance (|Coefficient|)', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    return fig
```

---

## 8. Support Vector Visualization

### 8.1 Support Vector Statistics

#### Function: `plot_support_vector_stats()`
```python
def plot_support_vector_stats(model, y_train):
    """
    Visualize support vector statistics

    Args:
        model: Trained SVM model
        y_train: Training labels

    Returns:
        matplotlib.figure.Figure: Support vector statistics
    """
    n_support = model.model.n_support_
    classes = model.classes_

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Support vectors per class
    ax1.bar(classes, n_support, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Number of Support Vectors', fontsize=12)
    ax1.set_title('Support Vectors per Class', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, v in enumerate(n_support):
        ax1.text(classes[i], v, str(v), ha='center', va='bottom', fontweight='bold')

    # Total distribution
    total_samples = len(y_train)
    total_sv = n_support.sum()
    non_sv = total_samples - total_sv

    labels = ['Support Vectors', 'Non-Support Vectors']
    sizes = [total_sv, non_sv]
    colors = ['#2ecc71', '#95a5a6']
    explode = (0.1, 0)

    ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90)
    ax2.set_title('Support Vector Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig
```

---

## 9. Comprehensive Dashboard

### 9.1 Create Results Dashboard

#### Function: `create_results_dashboard()`
```python
def create_results_dashboard(model, X_train, y_train, X_test, y_test,
                            metrics, cm, roc_data, feature_names=None):
    """
    Create comprehensive results dashboard

    Args:
        model: Trained model
        X_train, y_train: Training data
        X_test, y_test: Test data
        metrics: Performance metrics
        cm: Confusion matrix
        roc_data: ROC curve data
        feature_names: Feature names

    Returns:
        dict: Dictionary of all figures
    """
    figures = {}

    # 1. Decision boundary (2D)
    figures['decision_boundary_2d'] = plot_decision_boundary_2d(
        model, X_test, y_test, feature_names
    )

    # 2. Confusion matrix
    figures['confusion_matrix'] = plot_confusion_matrix(cm)

    # 3. ROC curve
    figures['roc_curve'] = plot_roc_curve(roc_data)

    # 4. Metrics bar chart
    figures['metrics_chart'] = plot_metrics_bar_chart(metrics)

    # 5. Support vector stats
    figures['support_vector_stats'] = plot_support_vector_stats(model, y_train)

    # 6. Feature importance (if applicable)
    if feature_names and hasattr(model.model, 'coef_'):
        figures['feature_importance'] = plot_feature_importance(
            model, feature_names
        )

    return figures
```

---

## 10. Utility Functions

### 10.1 Save Figure

#### Function: `save_figure()`
```python
def save_figure(fig, filename, dpi=300, format='png'):
    """
    Save matplotlib figure

    Args:
        fig: Matplotlib figure
        filename: Output filename
        dpi: Resolution
        format: File format
    """
    fig.savefig(filename, dpi=dpi, format=format, bbox_inches='tight')
```

---

### 10.2 Close All Figures

#### Function: `close_all_figures()`
```python
def close_all_figures():
    """Close all matplotlib figures"""
    plt.close('all')
```

---

## 11. Phase 4 Tasks & Subtasks

### Task 4.1: Process Flowchart
- [ ] Implement `create_process_flowchart()`
- [ ] Implement `create_interactive_flowchart()`
- [ ] Test with different current steps
- [ ] Verify highlighting works

### Task 4.2: Decision Boundary Plots
- [ ] Implement `plot_decision_boundary_2d()`
- [ ] Implement `plot_decision_boundary_3d()`
- [ ] Test with all kernel types
- [ ] Verify support vector highlighting

### Task 4.3: Confusion Matrix
- [ ] Implement `plot_confusion_matrix()`
- [ ] Implement `plot_confusion_matrix_interactive()`
- [ ] Test with binary and multi-class
- [ ] Verify normalization option

### Task 4.4: ROC Curves
- [ ] Implement `plot_roc_curve()`
- [ ] Implement `plot_roc_curve_interactive()`
- [ ] Test with binary and multi-class
- [ ] Verify AUC calculations

### Task 4.5: Performance Metrics
- [ ] Implement `plot_metrics_bar_chart()`
- [ ] Implement `plot_per_class_metrics()`
- [ ] Implement `plot_train_test_comparison()`
- [ ] Test all visualizations

### Task 4.6: Feature Importance
- [ ] Implement `plot_feature_importance()`
- [ ] Test with linear kernel
- [ ] Handle non-linear kernels

### Task 4.7: Support Vector Visualization
- [ ] Implement `plot_support_vector_stats()`
- [ ] Test with different datasets
- [ ] Verify counts are correct

### Task 4.8: Integration
- [ ] Implement `create_results_dashboard()`
- [ ] Test complete dashboard
- [ ] Verify all plots render correctly

---

## 12. Testing Phase 4 Completion

### Test 1: Flowchart Creation
```python
from src.visualizations import create_process_flowchart, create_interactive_flowchart

# Static flowchart
fig = create_process_flowchart(current_step=3)
assert fig is not None
print("✓ Static flowchart created")

# Interactive flowchart
fig_interactive = create_interactive_flowchart(current_step=3)
assert fig_interactive is not None
print("✓ Interactive flowchart created")
```

### Test 2: Decision Boundary
```python
from src.data_handler import load_sample_dataset, preprocess_pipeline
from src.svm_models import SVMClassifier
from src.visualizations import plot_decision_boundary_2d

# Train model
df = load_sample_dataset('medical')
data = preprocess_pipeline(df)
model = SVMClassifier(kernel_type='rbf')
model.fit(data['X_train'], data['y_train'])

# Plot
fig = plot_decision_boundary_2d(model, data['X_test'], data['y_test'])
assert fig is not None
print("✓ Decision boundary plotted")
```

### Test 3: All Visualizations
```python
from src.visualizations import (
    plot_confusion_matrix, plot_roc_curve,
    plot_metrics_bar_chart, plot_support_vector_stats
)
from src.svm_models import (
    calculate_accuracy_metrics, calculate_confusion_matrix,
    calculate_roc_curve
)

# Generate data
predictions = model.predict(data['X_test'])
probabilities = model.predict_proba(data['X_test'])

metrics = calculate_accuracy_metrics(data['y_test'], predictions)
cm = calculate_confusion_matrix(data['y_test'], predictions)
roc_data = calculate_roc_curve(data['y_test'], probabilities)

# Test all plots
fig1 = plot_confusion_matrix(cm)
fig2 = plot_roc_curve(roc_data)
fig3 = plot_metrics_bar_chart(metrics)
fig4 = plot_support_vector_stats(model, data['y_train'])

assert all([fig1, fig2, fig3, fig4])
print("✓ All visualizations working")
```

---

## 13. Common Issues & Solutions

### Issue 1: PCA warning for decision boundary
**Solution:** Ensure sufficient samples for dimensionality reduction

### Issue 2: Empty ROC curve
**Solution:** Check probability estimates are enabled in SVM

### Issue 3: Plotly not displaying
**Solution:** Use `fig.show()` or save as HTML

### Issue 4: Memory error with large plots
**Solution:** Reduce resolution parameter or sample data

### Issue 5: Feature importance not available
**Solution:** Only works with linear kernel

---

## 14. Phase 4 Completion Checklist

### Flowchart ✓
- [ ] Static flowchart implemented
- [ ] Interactive flowchart implemented
- [ ] Step highlighting working

### Decision Boundaries ✓
- [ ] 2D plot implemented
- [ ] 3D plot implemented
- [ ] Support vectors highlighted
- [ ] All kernels tested

### Confusion Matrix ✓
- [ ] Static heatmap implemented
- [ ] Interactive version implemented
- [ ] Normalization working

### ROC Curves ✓
- [ ] Static ROC curve implemented
- [ ] Interactive version implemented
- [ ] Multi-class handling working

### Performance Metrics ✓
- [ ] Bar chart implemented
- [ ] Per-class metrics implemented
- [ ] Train-test comparison implemented

### Feature Importance ✓
- [ ] Feature importance plot implemented
- [ ] Linear kernel tested

### Support Vectors ✓
- [ ] Statistics visualization implemented
- [ ] Distribution chart working

### Integration ✓
- [ ] Dashboard function implemented
- [ ] All plots integrated
- [ ] No errors or warnings

---

## 15. Next Steps

Once Phase 4 is complete:
1. Proceed to **Phase 5: Streamlit UI Integration**
2. Integrate all components into web interface
3. Create interactive user workflow

---

## 16. Time Tracking

**Estimated Time:** 5-6 hours
**Breakdown:**
- Flowchart: 30 minutes
- Decision boundaries: 1.5 hours
- Confusion matrix: 45 minutes
- ROC curves: 1 hour
- Performance metrics: 1 hour
- Feature importance: 30 minutes
- Support vectors: 30 minutes
- Testing & debugging: 45 minutes
- Buffer: 15-30 minutes

---

## Phase 4 Sign-Off

**Completed By:** ___________________
**Date:** ___________________
**Time Taken:** ___________________
**Issues Encountered:** ___________________
**Ready for Phase 5:** [ ] Yes [ ] No
