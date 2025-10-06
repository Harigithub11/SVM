# Phase 5: Streamlit UI Integration

## Overview
Create the complete Streamlit web application interface, integrating all previous modules (data handling, SVM models, visualizations) into an interactive user workflow.

**Estimated Duration:** 4-5 hours

**Prerequisites:** Phase 1, 2, 3, & 4 completed

---

## 1. Imports and Configuration

### Required Imports

```python
# File: src/app.py

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from data_handler import (
    load_dataset, load_sample_dataset, validate_dataset,
    detect_feature_types, detect_target_column,
    preprocess_pipeline
)
from svm_models import (
    SVMClassifier, train_svm_model,
    calculate_accuracy_metrics, generate_classification_report,
    calculate_confusion_matrix, calculate_roc_curve,
    generate_performance_summary
)
from visualizations import (
    create_interactive_flowchart, plot_decision_boundary_2d,
    plot_decision_boundary_3d, plot_confusion_matrix,
    plot_confusion_matrix_interactive, plot_roc_curve,
    plot_roc_curve_interactive, plot_metrics_bar_chart,
    plot_per_class_metrics, plot_support_vector_stats,
    plot_feature_importance
)

import warnings
warnings.filterwarnings('ignore')
```

---

### App Configuration

```python
# Page configuration
st.set_page_config(
    page_title="Classic SVM Application",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #2980b9;
        }
        .success-box {
            padding: 1rem;
            background-color: #d4edda;
            border-left: 5px solid #28a745;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .info-box {
            padding: 1rem;
            background-color: #d1ecf1;
            border-left: 5px solid #17a2b8;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .warning-box {
            padding: 1rem;
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
            border-radius: 5px;
            margin: 1rem 0;
        }
        h1 {
            color: #2c3e50;
            font-weight: bold;
        }
        h2 {
            color: #34495e;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
        }
        h3 {
            color: #7f8c8d;
        }
        </style>
    """, unsafe_allow_html=True)

load_css()
```

---

## 2. Session State Management

### Initialize Session State

```python
def initialize_session_state():
    """Initialize all session state variables"""
    if 'step' not in st.session_state:
        st.session_state.step = 1

    if 'domain' not in st.session_state:
        st.session_state.domain = None

    if 'kernel' not in st.session_state:
        st.session_state.kernel = None

    if 'dataset' not in st.session_state:
        st.session_state.dataset = None

    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None

    if 'model' not in st.session_state:
        st.session_state.model = None

    if 'results' not in st.session_state:
        st.session_state.results = None

    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None

initialize_session_state()
```

---

### Navigation Functions

```python
def next_step():
    """Move to next step"""
    st.session_state.step += 1

def previous_step():
    """Move to previous step"""
    if st.session_state.step > 1:
        st.session_state.step -= 1

def reset_app():
    """Reset entire application"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()
    st.rerun()
```

---

## 3. Domain Selection Interface

### Domain Configuration

```python
DOMAINS = {
    'medical': {
        'name': '🏥 Medical Diagnosis',
        'description': 'Predict diseases and medical conditions based on patient data',
        'icon': '🏥',
        'use_cases': [
            'Disease prediction',
            'Patient risk assessment',
            'Medical imaging classification'
        ],
        'sample_file': 'data/samples/medical_sample.csv'
    },
    'fraud': {
        'name': '💳 Fraud Detection',
        'description': 'Identify fraudulent transactions and activities',
        'icon': '💳',
        'use_cases': [
            'Credit card fraud detection',
            'Insurance fraud identification',
            'Anomaly detection'
        ],
        'sample_file': 'data/samples/fraud_sample.csv'
    },
    'classification': {
        'name': '📊 General Classification',
        'description': 'General-purpose pattern recognition and classification',
        'icon': '📊',
        'use_cases': [
            'Customer segmentation',
            'Image classification',
            'Text categorization'
        ],
        'sample_file': 'data/samples/classification_sample.csv'
    }
}
```

---

### Domain Selection UI

```python
def show_domain_selection():
    """Display domain selection interface"""
    st.title("🤖 Classic SVM Multi-Domain Application")

    st.markdown("---")
    st.header("Step 1: Select Application Domain")

    # Display flowchart
    flowchart = create_interactive_flowchart(current_step=1)
    st.plotly_chart(flowchart, use_container_width=True)

    st.markdown("---")

    # Create three columns for domain cards
    cols = st.columns(3)

    for idx, (domain_key, domain_info) in enumerate(DOMAINS.items()):
        with cols[idx]:
            st.markdown(f"""
                <div class="info-box">
                    <h2 style="text-align: center;">{domain_info['icon']}</h2>
                    <h3 style="text-align: center;">{domain_info['name']}</h3>
                    <p>{domain_info['description']}</p>
                    <ul>
                        {''.join([f'<li>{uc}</li>' for uc in domain_info['use_cases']])}
                    </ul>
                </div>
            """, unsafe_allow_html=True)

            if st.button(f"Select {domain_key.title()}", key=f"btn_{domain_key}"):
                st.session_state.domain = domain_key
                next_step()
                st.rerun()
```

---

## 4. Kernel Selection Interface

### Kernel Configuration

```python
KERNELS = {
    'linear': {
        'name': '📏 Linear Kernel',
        'description': 'Best for linearly separable data',
        'formula': 'K(x,y) = x·y',
        'icon': '📏',
        'when_to_use': [
            'High-dimensional data',
            'Text classification',
            'Sparse features'
        ],
        'parameters': {
            'C': 1.0
        }
    },
    'rbf': {
        'name': '🌀 RBF Kernel',
        'description': 'Handles non-linear decision boundaries',
        'formula': 'K(x,y) = exp(-γ||x-y||²)',
        'icon': '🌀',
        'when_to_use': [
            'Complex non-linear patterns',
            'Image classification',
            'General-purpose classification'
        ],
        'parameters': {
            'C': 1.0,
            'gamma': 'scale'
        }
    },
    'poly': {
        'name': '📐 Polynomial Kernel',
        'description': 'Creates curved decision boundaries',
        'formula': 'K(x,y) = (x·y + c)ᵈ',
        'icon': '📐',
        'when_to_use': [
            'Polynomial relationships',
            'Image processing',
            'Natural language processing'
        ],
        'parameters': {
            'C': 1.0,
            'degree': 3,
            'gamma': 'scale'
        }
    }
}
```

---

### Kernel Selection UI

```python
def show_kernel_selection():
    """Display kernel selection interface"""
    st.title("🤖 Classic SVM Multi-Domain Application")

    st.markdown(f"""
        <div class="success-box">
            <strong>Selected Domain:</strong> {DOMAINS[st.session_state.domain]['name']}
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.header("Step 2: Choose Kernel Type")

    # Display flowchart
    flowchart = create_interactive_flowchart(current_step=2)
    st.plotly_chart(flowchart, use_container_width=True)

    st.markdown("---")

    # Create three columns for kernel cards
    cols = st.columns(3)

    for idx, (kernel_key, kernel_info) in enumerate(KERNELS.items()):
        with cols[idx]:
            st.markdown(f"""
                <div class="info-box">
                    <h2 style="text-align: center;">{kernel_info['icon']}</h2>
                    <h3 style="text-align: center;">{kernel_info['name']}</h3>
                    <p><strong>Formula:</strong> {kernel_info['formula']}</p>
                    <p>{kernel_info['description']}</p>
                    <p><strong>Best for:</strong></p>
                    <ul>
                        {''.join([f'<li>{uc}</li>' for uc in kernel_info['when_to_use']])}
                    </ul>
                </div>
            """, unsafe_allow_html=True)

            if st.button(f"Select {kernel_key.title()}", key=f"btn_{kernel_key}"):
                st.session_state.kernel = kernel_key
                next_step()
                st.rerun()

    st.markdown("---")
    if st.button("⬅️ Back to Domain Selection"):
        previous_step()
        st.rerun()
```

---

## 5. Dataset Upload Interface

### Dataset Upload UI

```python
def show_dataset_upload():
    """Display dataset upload interface"""
    st.title("🤖 Classic SVM Multi-Domain Application")

    # Display selections
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="success-box">
                <strong>Domain:</strong> {DOMAINS[st.session_state.domain]['name']}
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="success-box">
                <strong>Kernel:</strong> {KERNELS[st.session_state.kernel]['name']}
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.header("Step 3: Upload Dataset")

    # Display flowchart
    flowchart = create_interactive_flowchart(current_step=3)
    st.plotly_chart(flowchart, use_container_width=True)

    st.markdown("---")

    # Upload options
    upload_option = st.radio(
        "Choose dataset source:",
        ["Use Sample Dataset", "Upload Custom CSV"],
        horizontal=True
    )

    if upload_option == "Use Sample Dataset":
        st.info(f"📁 Loading sample dataset for {st.session_state.domain} domain...")

        if st.button("Load Sample Dataset", type="primary"):
            try:
                df = load_sample_dataset(st.session_state.domain)
                st.session_state.dataset = df
                st.success("✅ Sample dataset loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading sample dataset: {str(e)}")

    else:
        st.info("📤 Upload your CSV file")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with your data. First row should contain column names."
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                # **NEW: Enhanced validation with comprehensive checks**
                validation = validate_uploaded_file(df, max_size_mb=10)

                # Display validation results
                if not validation['is_valid']:
                    st.error("❌ File validation failed:")
                    for error in validation['errors']:
                        st.error(error)
                else:
                    st.session_state.dataset = df
                    st.success("✅ File uploaded and validated successfully!")

                    # Show warnings if any
                    if validation['warnings']:
                        with st.expander("⚠️ Warnings (non-critical)", expanded=False):
                            for warning in validation['warnings']:
                                st.warning(warning)

                    # Show validation stats
                    with st.expander("📈 Validation Statistics", expanded=False):
                        stats = validation['stats']
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Total Rows", stats['rows'])
                            st.metric("Total Columns", stats['columns'])
                        with col_b:
                            st.metric("Memory (MB)", stats['memory_mb'])
                            st.metric("Missing (%)", stats['missing_pct'])
                        with col_c:
                            st.metric("Numeric Cols", stats['numeric_cols'])
                            st.metric("Duplicate Rows", stats['duplicate_rows'])

            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

    # Display dataset preview
    if st.session_state.dataset is not None:
        st.markdown("---")
        st.subheader("📊 Dataset Preview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", st.session_state.dataset.shape[0])
        with col2:
            st.metric("Columns", st.session_state.dataset.shape[1])
        with col3:
            st.metric("Missing Values", st.session_state.dataset.isnull().sum().sum())

        st.dataframe(st.session_state.dataset.head(10), use_container_width=True)

        # Data types
        with st.expander("🔍 View Data Types & Statistics"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Types:**")
                st.dataframe(st.session_state.dataset.dtypes, use_container_width=True)
            with col2:
                st.write("**Basic Statistics:**")
                st.dataframe(st.session_state.dataset.describe(), use_container_width=True)

        # Proceed button
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Kernel Selection"):
                previous_step()
                st.rerun()
        with col2:
            if st.button("➡️ Process & Train Model", type="primary"):
                next_step()
                st.rerun()
```

---

## 6. Processing Interface

### Processing & Training UI

```python
def show_processing():
    """Display processing and training interface"""
    st.title("🤖 Classic SVM Multi-Domain Application")

    st.markdown("---")
    st.header("Step 4: Processing & Training")

    # Display flowchart
    flowchart = create_interactive_flowchart(current_step=4)
    st.plotly_chart(flowchart, use_container_width=True)

    st.markdown("---")

    # **NEW: Modern progress tracking with st.status() (Streamlit 1.28+)**
    with st.status("🔄 Processing and Training Model...", expanded=True) as status:
        try:
            # Step 1: Preprocess data
            st.write("**1/4** 🔧 Preprocessing data...")

            preprocessed = preprocess_pipeline(
                st.session_state.dataset,
                target_column=None  # Auto-detect
            )
            st.session_state.preprocessed_data = preprocessed
            st.session_state.feature_names = preprocessed['feature_names']

            st.write("✅ Data preprocessing completed")

            # Step 2: Initialize model
            st.write("**2/4** 🤖 Initializing SVM model...")

            model = SVMClassifier(
                kernel_type=st.session_state.kernel,
                **KERNELS[st.session_state.kernel]['parameters']
            )

            st.write(f"✅ {KERNELS[st.session_state.kernel]['name']} initialized")

            # Step 3: Train model
            st.write("**3/4** 🎓 Training model...")

            model.fit(
                preprocessed['X_train'],
                preprocessed['y_train'],
                feature_names=preprocessed['feature_names']
            )
            st.session_state.model = model

            st.write(f"✅ Model trained in {model.training_time:.2f} seconds")

            # Step 4: Evaluate model
            st.write("**4/4** 📊 Evaluating performance...")

            results = generate_performance_summary(
                model,
                preprocessed['X_train'],
                preprocessed['y_train'],
                preprocessed['X_test'],
                preprocessed['y_test']
            )
            st.session_state.results = results

            # Update status to complete
            status.update(label="✅ Processing Complete!", state="complete", expanded=False)

        except Exception as e:
            status.update(label="❌ Processing Failed", state="error", expanded=True)
            st.error(f"Error during processing: {str(e)}")
            st.exception(e)

            if st.button("⬅️ Back to Dataset Upload"):
                previous_step()
                st.rerun()

    # Display quick metrics (outside status block)
    if 'results' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Quick Metrics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Test Accuracy",
                f"{st.session_state.results['test_metrics']['accuracy']:.2%}"
            )
        with col2:
            st.metric(
                "Precision",
                f"{st.session_state.results['test_metrics']['precision']:.2%}"
            )
        with col3:
            st.metric(
                "Recall",
                f"{st.session_state.results['test_metrics']['recall']:.2%}"
            )
        with col4:
            st.metric(
                "F1-Score",
                f"{st.session_state.results['test_metrics']['f1_score']:.2%}"
            )

        # Auto-advance button
        st.markdown("---")
        if st.button("➡️ View Detailed Results", type="primary"):
            next_step()
            st.rerun()
```

#### Why Use st.status()?

**Benefits of st.status() (Streamlit 1.28+) over progress bars:**
1. **Expandable container**: Shows/hides detailed progress steps
2. **State management**: Automatically displays icons for running/complete/error states
3. **Better UX**: Collapses when complete to save screen space
4. **Nested updates**: Can write multiple updates within the same container
5. **Error handling**: Built-in error state display with red icon

**States available:**
- `"running"` (default): Spinning icon, blue color
- `"complete"`: Green checkmark, collapses automatically
- `"error"`: Red X, stays expanded for debugging

---

## 7. Results Display Interface

### Results Dashboard UI

```python
def show_results():
    """Display comprehensive results dashboard"""
    st.title("🤖 Classic SVM Multi-Domain Application")

    st.markdown("---")
    st.header("Step 5: Results & Visualizations")

    # Display flowchart
    flowchart = create_interactive_flowchart(current_step=5)
    st.plotly_chart(flowchart, use_container_width=True)

    st.markdown("---")

    # Sidebar for navigation
    with st.sidebar:
        st.header("📑 Results Navigation")
        result_section = st.radio(
            "Jump to section:",
            [
                "Overview",
                "Performance Metrics",
                "Confusion Matrix",
                "ROC Curve",
                "Decision Boundary",
                "Support Vectors",
                "Feature Importance"
            ]
        )

    # Get data
    model = st.session_state.model
    data = st.session_state.preprocessed_data
    results = st.session_state.results

    # Calculate additional data
    predictions = model.predict(data['X_test'])
    probabilities = model.predict_proba(data['X_test'])
    cm = calculate_confusion_matrix(data['y_test'], predictions)
    roc_data = calculate_roc_curve(data['y_test'], probabilities)

    # Overview Section
    if result_section == "Overview":
        st.subheader("📋 Model Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Configuration**")
            st.write(f"- **Domain:** {DOMAINS[st.session_state.domain]['name']}")
            st.write(f"- **Kernel:** {KERNELS[st.session_state.kernel]['name']}")
            st.write(f"- **Dataset Size:** {st.session_state.dataset.shape}")
            st.write(f"- **Training Samples:** {len(data['X_train'])}")
            st.write(f"- **Test Samples:** {len(data['X_test'])}")

        with col2:
            st.markdown("**Model Information**")
            st.write(f"- **Training Time:** {model.training_time:.3f} seconds")
            st.write(f"- **Support Vectors:** {model.model.n_support_.sum()}")
            st.write(f"- **SV Ratio:** {model.model.n_support_.sum() / len(data['X_train']):.2%}")
            st.write(f"- **Number of Classes:** {len(model.classes_)}")

        st.markdown("---")

        # Overall metrics
        st.subheader("🎯 Overall Performance")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{results['test_metrics']['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{results['test_metrics']['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{results['test_metrics']['recall']:.2%}")
        with col4:
            st.metric("F1-Score", f"{results['test_metrics']['f1_score']:.2%}")

    # Performance Metrics Section
    elif result_section == "Performance Metrics":
        st.subheader("📊 Performance Metrics")

        # Metrics bar chart
        fig_metrics = plot_metrics_bar_chart(results['test_metrics'])
        st.pyplot(fig_metrics)

        # Per-class metrics
        st.markdown("---")
        st.subheader("📈 Per-Class Metrics")
        fig_per_class = plot_per_class_metrics(results['classification_report'])
        st.pyplot(fig_per_class)

        # Classification report table
        with st.expander("📄 Detailed Classification Report"):
            report_df = pd.DataFrame(results['classification_report']).transpose()
            st.dataframe(report_df, use_container_width=True)

    # Confusion Matrix Section
    elif result_section == "Confusion Matrix":
        st.subheader("🔲 Confusion Matrix")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Raw Counts**")
            fig_cm = plot_confusion_matrix(cm, normalize=False)
            st.pyplot(fig_cm)

        with col2:
            st.markdown("**Normalized**")
            fig_cm_norm = plot_confusion_matrix(cm, normalize=True,
                                               title='Confusion Matrix (Normalized)')
            st.pyplot(fig_cm_norm)

        # Interactive version
        st.markdown("---")
        st.markdown("**Interactive Confusion Matrix**")
        fig_cm_interactive = plot_confusion_matrix_interactive(cm)
        st.plotly_chart(fig_cm_interactive, use_container_width=True)

    # ROC Curve Section
    elif result_section == "ROC Curve":
        st.subheader("📈 ROC Curve")

        # Static ROC curve
        fig_roc = plot_roc_curve(roc_data)
        st.pyplot(fig_roc)

        # Interactive ROC curve
        st.markdown("---")
        st.markdown("**Interactive ROC Curve**")
        fig_roc_interactive = plot_roc_curve_interactive(roc_data)
        st.plotly_chart(fig_roc_interactive, use_container_width=True)

        # AUC scores
        with st.expander("📊 AUC Scores"):
            st.json(results['roc_auc'])

    # Decision Boundary Section
    elif result_section == "Decision Boundary":
        st.subheader("🎨 Decision Boundary Visualization")

        # 2D plot
        st.markdown("**2D Decision Boundary**")
        fig_2d = plot_decision_boundary_2d(
            model, data['X_test'], data['y_test'],
            feature_names=data['feature_names']
        )
        st.pyplot(fig_2d)

        # 3D plot
        st.markdown("---")
        st.markdown("**3D Decision Boundary**")
        fig_3d = plot_decision_boundary_3d(
            model, data['X_test'], data['y_test'],
            feature_names=data['feature_names']
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # Support Vectors Section
    elif result_section == "Support Vectors":
        st.subheader("🎯 Support Vector Analysis")

        fig_sv = plot_support_vector_stats(model, data['y_train'])
        st.pyplot(fig_sv)

        # Support vector details
        with st.expander("📋 Support Vector Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Support Vectors per Class:**")
                for i, n_sv in enumerate(model.model.n_support_):
                    st.write(f"- Class {model.classes_[i]}: {n_sv} SVs")
            with col2:
                st.write("**Statistics:**")
                st.write(f"- Total SVs: {model.model.n_support_.sum()}")
                st.write(f"- Percentage: {model.model.n_support_.sum() / len(data['X_train']):.2%}")

    # Feature Importance Section
    elif result_section == "Feature Importance":
        st.subheader("⭐ Feature Importance")

        if st.session_state.kernel == 'linear':
            fig_fi = plot_feature_importance(
                model, data['feature_names'], top_n=10
            )
            if fig_fi:
                st.pyplot(fig_fi)
            else:
                st.warning("Feature importance not available for this model")

            # Feature coefficients table
            with st.expander("📊 All Feature Coefficients"):
                coef_df = pd.DataFrame({
                    'Feature': data['feature_names'],
                    'Coefficient': model.model.coef_[0],
                    'Absolute': np.abs(model.model.coef_[0])
                }).sort_values('Absolute', ascending=False)
                st.dataframe(coef_df, use_container_width=True)
        else:
            st.info(f"ℹ️ Feature importance is only available for Linear kernel. "
                   f"Current kernel: {KERNELS[st.session_state.kernel]['name']}")

    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Start New Analysis"):
            reset_app()

    with col2:
        if st.button("⬅️ Back to Processing"):
            previous_step()
            st.rerun()

    with col3:
        # Download results button (placeholder)
        if st.button("💾 Download Report"):
            st.info("Download functionality coming soon!")
```

---

## 8. Main Application Flow

### Main App Function

```python
def main():
    """Main application function"""

    # Initialize session state
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100?text=SVM+Logo",
                use_column_width=True)

        st.markdown("---")
        st.header("ℹ️ About")
        st.write("""
        **Classic SVM Application**

        An interactive tool for Support Vector Machine classification
        across multiple domains.

        **Features:**
        - 3 Domain options
        - 3 Kernel types
        - Automated preprocessing
        - Comprehensive visualizations
        """)

        st.markdown("---")
        st.header("📍 Current Progress")
        st.write(f"**Step {st.session_state.step} of 5**")

        progress_labels = [
            "Domain Selection",
            "Kernel Selection",
            "Dataset Upload",
            "Processing",
            "Results"
        ]

        for i, label in enumerate(progress_labels, 1):
            if i < st.session_state.step:
                st.write(f"✅ {label}")
            elif i == st.session_state.step:
                st.write(f"▶️ {label}")
            else:
                st.write(f"⏸️ {label}")

        st.markdown("---")

        if st.session_state.step > 1:
            if st.button("🔄 Reset Application", use_container_width=True):
                reset_app()

    # Main content based on current step
    if st.session_state.step == 1:
        show_domain_selection()
    elif st.session_state.step == 2:
        show_kernel_selection()
    elif st.session_state.step == 3:
        show_dataset_upload()
    elif st.session_state.step == 4:
        show_processing()
    elif st.session_state.step == 5:
        show_results()

if __name__ == "__main__":
    main()
```

---

## 9. Error Handling & Validation

### Error Handling Functions

```python
def handle_errors(func):
    """Decorator for error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)
            return None
    return wrapper

def validate_uploaded_file(df, max_size_mb=10):
    """
    Enhanced file validation with comprehensive checks (2024 best practice)

    Args:
        df: Uploaded DataFrame
        max_size_mb: Maximum file size in MB

    Returns:
        dict: Validation results with errors, warnings, and stats
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    # **1. Basic structure checks**
    if df.empty:
        validation_result['errors'].append("❌ Dataset is empty")
        validation_result['is_valid'] = False

    if df.shape[0] < 50:
        validation_result['errors'].append(
            f"❌ Dataset too small: {df.shape[0]} rows (minimum 50 required)"
        )
        validation_result['is_valid'] = False

    if df.shape[1] < 2:
        validation_result['errors'].append(
            f"❌ Insufficient columns: {df.shape[1]} (minimum 2 required)"
        )
        validation_result['is_valid'] = False

    # **2. Data quality checks**
    # Columns with all missing values
    all_null_cols = df.columns[df.isnull().all()].tolist()
    if all_null_cols:
        validation_result['errors'].append(
            f"❌ Columns contain only missing values: {', '.join(all_null_cols)}"
        )
        validation_result['is_valid'] = False

    # High missing value percentage
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 50].index.tolist()
    if high_missing:
        validation_result['warnings'].append(
            f"⚠️ High missing values (>50%) in: {', '.join(high_missing)}"
        )

    # **3. Column type checks**
    # All object columns (no numeric data)
    if df.select_dtypes(include=[np.number]).shape[1] == 0:
        validation_result['warnings'].append(
            "⚠️ No numeric columns detected - ensure at least some numeric features exist"
        )

    # Duplicate column names
    if df.columns.duplicated().any():
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        validation_result['errors'].append(
            f"❌ Duplicate column names: {', '.join(dup_cols)}"
        )
        validation_result['is_valid'] = False

    # **4. Size and memory checks**
    memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    if memory_mb > max_size_mb:
        validation_result['errors'].append(
            f"❌ File too large: {memory_mb:.1f}MB (max {max_size_mb}MB)"
        )
        validation_result['is_valid'] = False

    # **5. Statistical checks**
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Zero variance columns
    zero_var_cols = []
    for col in numeric_cols:
        if df[col].nunique() == 1:
            zero_var_cols.append(col)

    if zero_var_cols:
        validation_result['warnings'].append(
            f"⚠️ Zero variance (constant) columns: {', '.join(zero_var_cols)}"
        )

    # **6. Collect statistics**
    validation_result['stats'] = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'memory_mb': round(memory_mb, 2),
        'numeric_cols': len(numeric_cols),
        'categorical_cols': df.select_dtypes(include=['object', 'category']).shape[1],
        'missing_pct': round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2),
        'duplicate_rows': df.duplicated().sum()
    }

    return validation_result
```

---

## 10. Phase 5 Tasks & Subtasks

### Task 5.1: App Configuration
- [ ] Set page configuration
- [ ] Create custom CSS
- [ ] Initialize session state
- [ ] Create navigation functions

### Task 5.2: Domain Selection
- [ ] Create domain configuration
- [ ] Implement domain selection UI
- [ ] Test domain selection flow
- [ ] Verify state management

### Task 5.3: Kernel Selection
- [ ] Create kernel configuration
- [ ] Implement kernel selection UI
- [ ] Test kernel selection flow
- [ ] Verify back button works

### Task 5.4: Dataset Upload
- [ ] Implement file upload widget
- [ ] Create sample dataset loader
- [ ] Add dataset preview
- [ ] Implement validation

### Task 5.5: Processing Interface
- [ ] Create progress tracking
- [ ] Implement preprocessing step
- [ ] Implement training step
- [ ] Implement evaluation step
- [ ] Display quick metrics

### Task 5.6: Results Display
- [ ] Create results navigation
- [ ] Implement overview section
- [ ] Implement metrics section
- [ ] Implement confusion matrix section
- [ ] Implement ROC curve section
- [ ] Implement decision boundary section
- [ ] Implement support vectors section
- [ ] Implement feature importance section

### Task 5.7: Integration & Testing
- [ ] Test complete workflow
- [ ] Test all three domains
- [ ] Test all three kernels
- [ ] Test error handling
- [ ] Test reset functionality

---

## 11. Testing Phase 5 Completion

### Test 1: App Launches
```bash
streamlit run src/app.py
# Expected: App opens in browser without errors
```

### Test 2: Complete Workflow
1. Select Medical domain
2. Select RBF kernel
3. Load sample dataset
4. Process and train
5. View all results sections
**Expected:** No errors, all visualizations display

### Test 3: All Combinations
Test all 9 combinations (3 domains × 3 kernels)
**Expected:** All work correctly

### Test 4: Error Handling
- Upload invalid CSV
- Upload empty file
- Try processing without dataset
**Expected:** Appropriate error messages

### Test 5: Navigation
- Test back buttons
- Test reset button
- Test step progression
**Expected:** Smooth navigation

---

## 12. Common Issues & Solutions

### Issue 1: Streamlit module not found
**Solution:** `pip install streamlit`

### Issue 2: Plots not displaying
**Solution:** Use `st.pyplot()` and `st.plotly_chart()`

### Issue 3: Session state not persisting
**Solution:** Ensure `st.rerun()` called after state changes

### Issue 4: File upload not working
**Solution:** Check file path and permissions

### Issue 5: CSS not applied
**Solution:** Use `unsafe_allow_html=True`

---

## 13. Phase 5 Completion Checklist

### Configuration ✓
- [ ] Page config set
- [ ] Custom CSS applied
- [ ] Session state initialized

### Domain Selection ✓
- [ ] UI implemented
- [ ] All domains selectable
- [ ] Flowchart displays

### Kernel Selection ✓
- [ ] UI implemented
- [ ] All kernels selectable
- [ ] Back button works

### Dataset Upload ✓
- [ ] File upload working
- [ ] Sample datasets loading
- [ ] Preview displaying
- [ ] Validation working

### Processing ✓
- [ ] Progress tracking working
- [ ] Preprocessing executing
- [ ] Training executing
- [ ] Metrics displaying

### Results ✓
- [ ] All sections implemented
- [ ] All visualizations displaying
- [ ] Navigation working
- [ ] Data correct

### Integration ✓
- [ ] Complete workflow tested
- [ ] All combinations tested
- [ ] Error handling working
- [ ] No critical bugs

---

## 14. Deployment (Optional)

### Streamlit Cloud Deployment

```bash
# 1. Create requirements.txt with all dependencies
# 2. Push to GitHub
# 3. Connect to Streamlit Cloud
# 4. Deploy from repository
```

### Local Deployment

```bash
# Run locally
streamlit run src/app.py

# Run with custom port
streamlit run src/app.py --server.port 8080

# Run with external access
streamlit run src/app.py --server.address 0.0.0.0
```

---

## 15. Next Steps & Enhancements

### Possible Enhancements:
1. **Hyperparameter Tuning UI**
   - Manual parameter adjustment
   - Grid search option

2. **Model Comparison**
   - Compare all kernels side-by-side
   - Best model recommendation

3. **Export Options**
   - Download trained model
   - Export results as PDF
   - Save visualizations

4. **Advanced Features**
   - Cross-validation
   - Feature engineering
   - Custom preprocessing

---

## 16. Time Tracking

**Estimated Time:** 4-5 hours
**Breakdown:**
- Configuration & setup: 30 minutes
- Domain selection: 30 minutes
- Kernel selection: 30 minutes
- Dataset upload: 45 minutes
- Processing interface: 45 minutes
- Results dashboard: 1.5 hours
- Testing & debugging: 45 minutes
- Buffer: 15-30 minutes

---

## 17. Final Project Completion

### All Phases Complete ✓
- [ ] Phase 1: Setup complete
- [ ] Phase 2: Data & preprocessing complete
- [ ] Phase 3: SVM implementation complete
- [ ] Phase 4: Visualizations complete
- [ ] Phase 5: Streamlit UI complete

### Final Testing ✓
- [ ] End-to-end workflow tested
- [ ] All features working
- [ ] No critical bugs
- [ ] Documentation complete

### Production Ready ✓
- [ ] Code clean and documented
- [ ] README updated
- [ ] Requirements.txt accurate
- [ ] Ready for deployment

---

## Phase 5 Sign-Off

**Completed By:** ___________________
**Date:** ___________________
**Time Taken:** ___________________
**Issues Encountered:** ___________________
**Project Complete:** [ ] Yes [ ] No

---

## 18. Final Launch Command

```bash
# Launch the application
streamlit run src/app.py
```

**🎉 Congratulations! Your Classic SVM Multi-Domain Application is complete!**
