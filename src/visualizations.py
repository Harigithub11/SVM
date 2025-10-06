"""
Visualizations Module
Comprehensive visualization functions for SVM analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


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


def create_simple_flowchart_fallback(current_step=None):
    """
    Fallback flowchart using matplotlib (no recursion issues)

    Args:
        current_step (int): Current step to highlight (1-5)

    Returns:
        matplotlib.figure.Figure: Simple flowchart
    """
    steps = ['Select\nDomain', 'Choose\nKernel', 'Upload\nDataset', 'Process &\nTrain', 'View\nResults']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

    fig, ax = plt.subplots(figsize=(14, 2))

    for i, (step, color) in enumerate(zip(steps, colors)):
        opacity = 1.0 if current_step is None or current_step == i + 1 else 0.3
        # Draw box
        rect = plt.Rectangle((i * 2.5, 0), 2, 1, facecolor=color, alpha=opacity, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        # Add text
        ax.text(i * 2.5 + 1, 0.5, step, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        # Add arrow
        if i < len(steps) - 1:
            ax.arrow(i * 2.5 + 2.1, 0.5, 0.3, 0, head_width=0.2, head_length=0.1, fc='black', ec='black', alpha=opacity)

    ax.set_xlim(-0.5, len(steps) * 2.5)
    ax.set_ylim(-0.3, 1.3)
    ax.axis('off')
    ax.set_title('SVM Application Workflow', fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    return fig


def create_interactive_flowchart(current_step=None):
    """
    Create interactive flowchart using Plotly (with fallback to matplotlib)

    Args:
        current_step (int): Current step to highlight (1-5)

    Returns:
        plotly.graph_objects.Figure or matplotlib.figure.Figure: Interactive flowchart
    """
    try:
        steps = ['Select Domain', 'Choose Kernel', 'Upload Dataset', 'Process & Train', 'View Results']
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

        # Create figure
        fig = go.Figure()

        # Add boxes
        for i, (step, color) in enumerate(zip(steps, colors)):
            opacity = 1.0 if current_step is None or current_step == i + 1 else 0.3

            fig.add_trace(go.Scatter(
                x=[i],
                y=[0],
                mode='markers+text',
                marker=dict(size=80, color=color, opacity=opacity, symbol='square'),
                text=step,
                textposition='middle center',
                textfont=dict(color='white', size=10, family='Arial Black'),
                showlegend=False,
                hoverinfo='text',
                hovertext=f'Step {i+1}: {step}'
            ))

        # Add arrows
        for i in range(len(steps) - 1):
            opacity = 1.0 if current_step is None or current_step == i + 1 else 0.3
            fig.add_annotation(
                x=i + 0.5,
                y=0,
                ax=i,
                ay=0,
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
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
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.5, 0.5]),
            height=200,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return fig

    except RecursionError:
        # Fallback to matplotlib if Plotly has recursion issues
        return create_simple_flowchart_fallback(current_step)


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=None,
                         title='Confusion Matrix', cmap='Blues'):
    """
    Plot confusion matrix using sklearn's ConfusionMatrixDisplay (2024 best practice)

    Args:
        y_true: True labels
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

    # Use ConfusionMatrixDisplay from sklearn 1.7.2
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        normalize=normalize,
        cmap=cmap,
        ax=ax,
        colorbar=True,
        values_format=None
    )

    # Customize labels
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    ax.grid(False)
    plt.tight_layout()
    return fig


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


def plot_roc_curve(roc_data, title='ROC Curve'):
    """
    Plot ROC curve (supports binary and multi-class with micro/macro)

    Args:
        roc_data: ROC data from calculate_roc_curve()
        title: Plot title

    Returns:
        matplotlib.figure.Figure: ROC curve figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if 'binary' in roc_data:
        # Binary classification
        ax.plot(roc_data['binary']['fpr'], roc_data['binary']['tpr'],
               label=f"ROC (AUC = {roc_data['binary']['auc']:.3f})",
               linewidth=2)
    else:
        # Multi-class with micro/macro
        # Plot micro-average
        if 'micro' in roc_data:
            ax.plot(roc_data['micro']['fpr'], roc_data['micro']['tpr'],
                   label=f"Micro-average (AUC = {roc_data['micro']['auc']:.3f})",
                   linewidth=3, linestyle='--', color='deeppink')

        # Plot macro-average
        if 'macro' in roc_data:
            ax.plot(roc_data['macro']['fpr'], roc_data['macro']['tpr'],
                   label=f"Macro-average (AUC = {roc_data['macro']['auc']:.3f})",
                   linewidth=3, linestyle='--', color='navy')

        # Plot per-class curves
        colors = plt.cm.Set1(np.linspace(0, 1, len([k for k in roc_data.keys() if k.startswith('class_')])))
        for i, (key, color) in enumerate(zip([k for k in roc_data.keys() if k.startswith('class_')], colors)):
            ax.plot(roc_data[key]['fpr'], roc_data[key]['tpr'],
                   label=f"Class {roc_data[key]['class_label']} (AUC = {roc_data[key]['auc']:.3f})",
                   linewidth=2, color=color)

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_roc_curve_interactive(roc_data):
    """
    Create interactive ROC curve using Plotly

    Args:
        roc_data: ROC data from calculate_roc_curve()

    Returns:
        plotly.graph_objects.Figure: Interactive ROC curve
    """
    fig = go.Figure()

    if 'binary' in roc_data:
        # Binary classification
        fig.add_trace(go.Scatter(
            x=roc_data['binary']['fpr'],
            y=roc_data['binary']['tpr'],
            mode='lines',
            name=f"ROC (AUC = {roc_data['binary']['auc']:.3f})",
            line=dict(width=2)
        ))
    else:
        # Multi-class
        if 'micro' in roc_data:
            fig.add_trace(go.Scatter(
                x=roc_data['micro']['fpr'],
                y=roc_data['micro']['tpr'],
                mode='lines',
                name=f"Micro-avg (AUC = {roc_data['micro']['auc']:.3f})",
                line=dict(width=3, dash='dash')
            ))

        if 'macro' in roc_data:
            fig.add_trace(go.Scatter(
                x=roc_data['macro']['fpr'],
                y=roc_data['macro']['tpr'],
                mode='lines',
                name=f"Macro-avg (AUC = {roc_data['macro']['auc']:.3f})",
                line=dict(width=3, dash='dash')
            ))

        for key in [k for k in roc_data.keys() if k.startswith('class_')]:
            fig.add_trace(go.Scatter(
                x=roc_data[key]['fpr'],
                y=roc_data[key]['tpr'],
                mode='lines',
                name=f"Class {roc_data[key]['class_label']} (AUC = {roc_data[key]['auc']:.3f})"
            ))

    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='gray')
    ))

    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=600,
        width=800,
        hovermode='closest'
    )

    return fig


def plot_metrics_bar_chart(metrics, title='Model Performance Metrics'):
    """
    Plot metrics as bar chart

    Args:
        metrics: Dictionary of metrics
        title: Plot title

    Returns:
        matplotlib.figure.Figure: Bar chart figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    bars = ax.bar(metric_names, metric_values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


def plot_decision_boundary_2d(X, y, model, feature_names=None, title='Decision Boundary'):
    """
    Plot 2D decision boundary using PCA if needed

    Args:
        X: Features (will use PCA if > 2D)
        y: Labels
        model: Trained model
        feature_names: Feature names
        title: Plot title

    Returns:
        matplotlib.figure.Figure: Decision boundary plot
    """
    # Reduce to 2D using PCA if needed
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        xlabel = 'First Principal Component'
        ylabel = 'Second Principal Component'
    else:
        X_2d = X
        xlabel = feature_names[0] if feature_names else 'Feature 1'
        ylabel = feature_names[1] if feature_names else 'Feature 2'

    # Create mesh
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on mesh
    if X.shape[1] > 2:
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_full = pca.inverse_transform(mesh_points)
        Z = model.predict(mesh_full)
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis',
                        edgecolors='black', s=50, alpha=0.8)

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Class')
    plt.tight_layout()

    return fig


def plot_decision_boundary_3d(X, y, model, feature_names=None):
    """
    Plot 3D decision boundary using Plotly

    Args:
        X: Features (will use PCA if > 3D)
        y: Labels
        model: Trained model
        feature_names: Feature names

    Returns:
        plotly.graph_objects.Figure: 3D decision boundary plot
    """
    # Reduce to 3D using PCA if needed
    if X.shape[1] > 3:
        pca = PCA(n_components=3)
        X_3d = pca.fit_transform(X)
        labels = ['PC1', 'PC2', 'PC3']
    else:
        X_3d = X[:, :3]
        labels = feature_names[:3] if feature_names else ['F1', 'F2', 'F3']

    # Create 3D scatter
    fig = go.Figure(data=[go.Scatter3d(
        x=X_3d[:, 0],
        y=X_3d[:, 1],
        z=X_3d[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=y,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Class')
        ),
        text=[f'Class: {label}' for label in y],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='3D Decision Boundary Visualization',
        scene=dict(
            xaxis_title=labels[0],
            yaxis_title=labels[1],
            zaxis_title=labels[2]
        ),
        height=700,
        width=900
    )

    return fig


def plot_svm_hyperplane_2d(X, y, model, X_train=None, feature_names=None, title='SVM Hyperplane & Support Vectors'):
    """
    Plot SVM hyperplane with support vectors highlighted (2D visualization)
    Shows the decision boundary, margins, and support vectors like a textbook diagram

    Args:
        X: Test features for plotting data points (will use first 2 or PCA if >2D)
        y: Test labels
        model: Trained SVM model
        X_train: Training features (needed to show support vectors from training)
        feature_names: Feature names
        title: Plot title

    Returns:
        matplotlib.figure.Figure: Hyperplane visualization
    """
    # If no training data provided, just use test data
    if X_train is None:
        X_train = X

    # Reduce to 2D if needed
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        X_train_2d = pca.transform(X_train) if X_train is not X else X_2d
        xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)'
        ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)'
    else:
        X_2d = X
        X_train_2d = X_train
        xlabel = feature_names[0] if feature_names else 'Feature 1'
        ylabel = feature_names[1] if feature_names and len(feature_names) > 1 else 'Feature 2'

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on mesh
    if X.shape[1] > 2:
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_full = pca.inverse_transform(mesh_points)
        Z = model.predict(mesh_full)
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu', levels=20)

    # Plot decision boundary line
    ax.contour(xx, yy, Z, colors='black', linewidths=2, levels=[0.5])

    # Get unique classes and colors
    classes = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(classes)))

    # Plot data points (test data)
    for idx, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                  c=[colors[idx]], label=f'Class {cls}',
                  edgecolors='black', s=80, alpha=0.7, linewidth=1)

    # Highlight support vectors (from training data)
    if hasattr(model.model, 'support_') and X_train_2d is not None:
        support_vectors_2d = X_train_2d[model.model.support_]
        ax.scatter(support_vectors_2d[:, 0], support_vectors_2d[:, 1],
                  s=250, linewidths=3, facecolors='none',
                  edgecolors='lime', label='Support Vectors',
                  marker='o', alpha=0.9)

        # Add annotation
        n_sv = len(model.model.support_)
        ax.text(0.02, 0.98, f'Support Vectors: {n_sv}',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()

    return fig


def plot_svm_hyperplane_with_margins(X, y, model, X_train=None, y_train=None, feature_names=None,
                                     title='SVM Hyperplane with Margins'):
    """
    Plot classic textbook-style SVM diagram with hyperplane and margin lines
    Shows the optimal separating hyperplane with margin boundaries clearly marked

    Args:
        X: Test data features (will be reduced to 2D if needed)
        y: Test data labels
        model: Trained SVM model
        X_train: Training data (needed for support vectors)
        feature_names: List of feature names
        title: Plot title

    Returns:
        matplotlib.figure.Figure: Hyperplane visualization with margins
    """
    # Use training data if not provided
    if X_train is None:
        X_train = X
    if y_train is None:
        y_train = y

    # Reduce to 2D if needed and retrain a 2D model for visualization
    needs_pca = X.shape[1] > 2

    if needs_pca:
        from sklearn.svm import SVC
        pca = PCA(n_components=2)

        # Combine train and test for PCA fitting
        X_combined = np.vstack([X_train, X])
        pca.fit(X_combined)

        X_2d = pca.transform(X)
        X_train_2d = pca.transform(X_train)

        # Train a new 2D SVM just for visualization
        viz_model = SVC(kernel=model.model.get_params()['kernel'],
                       C=model.model.get_params().get('C', 1.0),
                       gamma=model.model.get_params().get('gamma', 'scale'))

        viz_model.fit(X_train_2d, y_train)

        xlabel = 'First Principal Component'
        ylabel = 'Second Principal Component'
    else:
        X_2d = X
        X_train_2d = X_train
        viz_model = model.model  # Use original model for 2D data
        xlabel = feature_names[0] if feature_names else 'Feature 1'
        ylabel = feature_names[1] if feature_names and len(feature_names) > 1 else 'Feature 2'

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get unique classes
    classes = np.unique(y)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, teal, blue

    # Plot data points
    for idx, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                  c=[colors[idx % len(colors)]], label=f'Class {cls}',
                  edgecolors='black', s=100, alpha=0.8, linewidth=1.5)

    # Create mesh for decision boundary and margins
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Detect binary vs multi-class classification
    n_classes = len(np.unique(y_train))

    try:
        if n_classes == 2:
            # BINARY CLASSIFICATION: Use decision_function to show hyperplane + margins
            if hasattr(viz_model, 'decision_function'):
                Z = viz_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                # Plot decision boundary (hyperplane) - thick black line at Z=0
                ax.contour(xx, yy, Z, colors='black', levels=[0], linewidths=3,
                          linestyles='solid', label='Decision Boundary')

                # Plot margins (dashed lines at Z=-1 and Z=1)
                ax.contour(xx, yy, Z, colors='black', levels=[-1, 1], linewidths=2,
                          linestyles='dashed', alpha=0.7)

                # Add text annotation for margins
                ax.text(0.05, 0.95, 'Margin', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                # Fallback: use predict
                Z = viz_model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                ax.contour(xx, yy, Z, colors='black', linewidths=3, linestyles='solid')

        else:
            # MULTI-CLASS CLASSIFICATION: Use predict() to show class regions
            # (Margins don't exist for multi-class SVM!)
            Z = viz_model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Show class regions with light colors
            ax.contourf(xx, yy, Z, alpha=0.15, cmap='Set1', levels=np.arange(n_classes+1)-0.5)

            # Show boundaries between classes (thick black lines)
            ax.contour(xx, yy, Z, colors='black', linewidths=2,
                      linestyles='solid', alpha=0.8, levels=np.arange(n_classes+1)-0.5)

            # Add text annotation for class regions
            ax.text(0.05, 0.95, 'Class Regions', transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    except Exception as e:
        # Fallback: Just show data points without decision boundary
        ax.text(0.5, 0.5, f'Decision boundary could not be plotted\n(Error: {str(e)[:40]}...)',
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Highlight support vectors with large circles
    if hasattr(viz_model, 'support_'):
        support_vectors_2d = X_train_2d[viz_model.support_]
        ax.scatter(support_vectors_2d[:, 0], support_vectors_2d[:, 1],
                  s=300, linewidths=3, facecolors='none',
                  edgecolors='#00FF00', label='Support Vectors',
                  marker='o', alpha=1.0, zorder=10)

        # Add text box with SV count
        n_sv = len(viz_model.support_)
        textstr = f'Support Vectors: {n_sv}'
        props = dict(boxstyle='round', facecolor='yellow', alpha=0.8)
        ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='bottom', fontweight='bold', bbox=props)

    # Add labels showing "Hyperplane" with arrow
    ax.annotate('Hyperplane\n(Decision Boundary)',
                xy=(0.5, 0.5), xycoords='axes fraction',
                xytext=(0.7, 0.85), textcoords='axes fraction',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                              color='red', lw=2))

    ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot feature importance (for linear kernels)

    Args:
        model: Trained SVM model
        feature_names: List of feature names
        top_n: Number of top features to show

    Returns:
        matplotlib.figure.Figure: Feature importance plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if hasattr(model.model, 'coef_'):
        # Get coefficients
        coef = np.abs(model.model.coef_).mean(axis=0)

        # Sort by importance
        indices = np.argsort(coef)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_values = coef[indices]

        # Plot
        bars = ax.barh(range(len(top_features)), top_values, color='steelblue')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Absolute Coefficient Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.3f}',
                   ha='left', va='center', fontweight='bold')

        plt.tight_layout()
    else:
        ax.text(0.5, 0.5, 'Feature importance not available\nfor non-linear kernels',
               ha='center', va='center', fontsize=14)
        ax.axis('off')

    return fig


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

    ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.8, edgecolor='black')
    ax.bar(x, recall, width, label='Recall', color='#2ecc71', alpha=0.8, edgecolor='black')
    ax.bar(x + width, f1, width, label='F1-Score', color='#f39c12', alpha=0.8, edgecolor='black')

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()

    return fig


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

    bars1 = ax.bar(x - width/2, train_values, width, label='Training',
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, test_values, width, label='Testing',
                   color='#e74c3c', alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Training vs Testing Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.15)
    plt.tight_layout()

    return fig


def plot_support_vector_stats(model, y_train):
    """
    Visualize support vector statistics

    Args:
        model: Trained SVM model
        y_train: Training labels

    Returns:
        matplotlib.figure.Figure: Support vector statistics
    """
    if not hasattr(model.model, 'n_support_'):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, 'Support vector information not available\nfor this model type',
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    n_support = model.model.n_support_
    classes = model.classes_

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Support vectors per class
    bars = ax1.bar(range(len(classes)), n_support, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Support Vectors', fontsize=12, fontweight='bold')
    ax1.set_title('Support Vectors per Class', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels([f'Class {c}' for c in classes])
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                str(int(height)), ha='center', va='bottom', fontweight='bold')

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
    ax2.set_title(f'Support Vector Distribution\n(Total: {total_samples} samples)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def create_results_dashboard(model, X_train, y_train, X_test, y_test,
                            metrics, cm, roc_data, feature_names=None,
                            class_names=None):
    """
    Create comprehensive results dashboard

    Args:
        model: Trained model
        X_train, y_train: Training data
        X_test, y_test: Test data
        metrics: Performance metrics dictionary
        cm: Confusion matrix
        roc_data: ROC curve data
        feature_names: Feature names
        class_names: Class names

    Returns:
        dict: Dictionary of all figures
    """
    figures = {}

    # 1. Decision boundary (2D)
    try:
        figures['decision_boundary_2d'] = plot_decision_boundary_2d(
            X_test, y_test, model, feature_names
        )
    except Exception as e:
        print(f"Warning: Could not create 2D decision boundary: {e}")

    # 2. Decision boundary (3D)
    try:
        figures['decision_boundary_3d'] = plot_decision_boundary_3d(
            X_test, y_test, model, feature_names
        )
    except Exception as e:
        print(f"Warning: Could not create 3D decision boundary: {e}")

    # 3. Confusion matrix
    try:
        y_pred = model.predict(X_test)
        figures['confusion_matrix'] = plot_confusion_matrix(
            y_test, y_pred, class_names=class_names
        )
    except Exception as e:
        print(f"Warning: Could not create confusion matrix: {e}")

    # 4. ROC curve
    try:
        figures['roc_curve'] = plot_roc_curve(roc_data)
    except Exception as e:
        print(f"Warning: Could not create ROC curve: {e}")

    # 5. Metrics bar chart
    try:
        figures['metrics_chart'] = plot_metrics_bar_chart(metrics)
    except Exception as e:
        print(f"Warning: Could not create metrics chart: {e}")

    # 6. Support vector stats
    try:
        figures['support_vector_stats'] = plot_support_vector_stats(model, y_train)
    except Exception as e:
        print(f"Warning: Could not create support vector stats: {e}")

    # 7. Feature importance (if applicable)
    if feature_names and hasattr(model.model, 'coef_'):
        try:
            figures['feature_importance'] = plot_feature_importance(
                model, feature_names
            )
        except Exception as e:
            print(f"Warning: Could not create feature importance: {e}")

    return figures
