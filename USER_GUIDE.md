# Classic SVM Multi-Domain Application - User Guide

## Quick Start

### Starting the Application

```bash
# Navigate to project directory
cd "C:\Hari\SRM\7th Sem\QML\Classic SVM"

# Activate virtual environment
venv\Scripts\activate

# Run the application
streamlit run src/app.py
```

The application will open in your default browser at **http://localhost:8501**

---

## Step-by-Step Usage

### Step 1: Select Domain

Choose one of three application domains:

- **🏥 Medical Diagnosis** - Disease prediction and medical classification
- **💳 Fraud Detection** - Transaction fraud identification
- **📊 General Classification** - Pattern recognition and data classification

Click the button under your chosen domain to proceed.

---

### Step 2: Choose Kernel Type

Select an SVM kernel based on your data characteristics:

- **📏 Linear Kernel**
  - Best for: High-dimensional data, text classification
  - Formula: K(x,y) = x·y
  - Use when: Data is linearly separable

- **🌀 RBF Kernel**
  - Best for: Complex non-linear patterns, general-purpose
  - Formula: K(x,y) = exp(-γ||x-y||²)
  - Use when: Unknown data structure or non-linear patterns

- **📐 Polynomial Kernel**
  - Best for: Polynomial relationships, image processing
  - Formula: K(x,y) = (x·y + c)ᵈ
  - Use when: Data has polynomial relationships

Click the button under your chosen kernel to proceed.

---

### Step 3: Upload Dataset

**Option A: Use Sample Dataset**
1. Click "Load Sample Dataset" button
2. Pre-loaded domain-specific dataset will be used
3. Preview appears automatically

**Option B: Upload Custom CSV**
1. Click "Browse files" button
2. Select your CSV file (must have headers)
3. File validation will run automatically
4. Review any warnings or errors
5. Check the dataset preview

**Dataset Requirements:**
- Format: CSV with header row
- Minimum: 50 rows, 2 columns
- Maximum size: 10 MB
- Should include a target/label column

Click "Process & Train Model" when ready.

---

### Step 4: Processing & Training

Watch the real-time progress as your model is:

1. **Preprocessing data** - Handling missing values, encoding, scaling
2. **Initializing SVM** - Setting up the chosen kernel
3. **Training model** - Fitting on training data
4. **Evaluating performance** - Testing accuracy and metrics

The status container will collapse when complete, showing quick metrics:
- Test Accuracy
- Precision
- Recall
- F1-Score

Click "View Detailed Results" to see comprehensive analysis.

---

### Step 5: View Results

Use the **sidebar navigation** to explore different result sections:

#### 📋 Overview
- Configuration summary
- Model information
- Overall performance metrics
- Training statistics

#### 📊 Performance Metrics
- Bar charts of accuracy, precision, recall, F1
- Per-class performance breakdown
- Training vs Testing comparison
- Detailed classification report

#### 🔲 Confusion Matrix
- Raw counts matrix
- Normalized matrix (by true labels)
- Interactive Plotly version
- Per-class accuracy visualization

#### 📈 ROC Curve
- Static ROC curve with AUC scores
- Interactive Plotly version
- Multi-class support (micro/macro averaging)
- Per-class ROC curves

#### 🎨 Decision Boundary
- 2D visualization (using PCA if needed)
- 3D interactive visualization
- Support vectors highlighted
- Color-coded by class

#### 🎯 Support Vectors
- Count per class
- Distribution pie chart
- Support vector ratio
- Detailed statistics

#### ⭐ Feature Importance
- **Only for Linear kernel**
- Top 10 most important features
- Bar chart with coefficients
- Complete feature ranking table

---

## Tips & Best Practices

### Choosing a Kernel

**Use Linear when:**
- You have many features (>1000)
- Data is text-based
- You need feature importance insights
- Training speed is critical

**Use RBF when:**
- You're unsure about data structure
- Data has complex patterns
- You have moderate sample size
- Non-linear boundaries expected

**Use Polynomial when:**
- You know data has polynomial relationships
- Working with image data
- Experimenting with curved boundaries
- Have enough samples (polynomial can overfit)

---

### Dataset Preparation

**Good CSV format:**
```csv
age,blood_pressure,cholesterol,bmi,smoker,diagnosis
45,120,180,24.5,no,healthy
67,145,240,28.3,yes,at_risk
```

**Requirements:**
- First row must be column names
- Last column usually is the target/label
- No missing headers
- Consistent data types per column
- Minimal missing values (<30%)

---

### Interpreting Results

**High Accuracy (>95%):**
- Model performing excellently
- Check for data leakage
- Verify with confusion matrix

**Medium Accuracy (80-95%):**
- Good model performance
- Review per-class metrics
- Consider feature engineering

**Low Accuracy (<80%):**
- Try different kernel
- Check data quality
- May need more features or samples

**Support Vector Ratio:**
- Low (2-10%): Clean, separable data
- Medium (10-30%): Moderate complexity
- High (>30%): Complex or noisy data

---

## Troubleshooting

### "File validation failed"
- Check file size < 10MB
- Ensure at least 50 rows, 2 columns
- Verify CSV format with proper headers

### "Processing failed"
- Check for all-missing columns
- Ensure at least 2 classes in target
- Verify numeric/categorical data types

### "Feature importance not available"
- Only works with Linear kernel
- Switch to Linear kernel to see this

### Visualizations not showing
- Refresh the page
- Check browser console for errors
- Try different result section

---

## Navigation Buttons

**Throughout the app:**
- **⬅️ Back** - Return to previous step
- **➡️ Next/Process** - Advance to next step
- **🔄 Reset Application** - Start over from Step 1
- **🔄 Start New Analysis** - Reset from results page

---

## Sample Datasets

All sample datasets are pre-validated and ready to use:

- **Medical:** 500 patients, 8 features
- **Fraud:** 600 transactions, 8 features
- **Classification:** 350 samples, 9 features

Perfect for testing and understanding the application.

---

## Advanced Features

### Session State
- Your selections are preserved as you navigate
- Can go back/forward without losing data
- Reset completely with Reset button

### Modern Preprocessing
- Uses ColumnTransformer (prevents data leakage)
- Automatic feature type detection
- Stratified train-test split
- Handles mixed data types

### Visualization Export
- Right-click on Plotly charts → "Download plot as PNG"
- Matplotlib charts can be screenshot
- Classification report can be copied

---

## Keyboard Shortcuts

- **Tab** - Navigate between buttons
- **Enter** - Click focused button
- **Ctrl+R** - Refresh page
- **F11** - Fullscreen mode

---

## Support & Resources

**Documentation:**
- `/docs` folder contains detailed phase guides
- `README.md` for project overview
- `TEST_REPORT.md` for comprehensive testing results

**Common Issues:**
- Check `docs/CRITICAL_UPDATES_REQUIRED.md` for known issues
- Review test files in `/tests` for examples

---

## Example Workflows

### Workflow 1: Quick Demo
1. Select **Medical** domain
2. Choose **Linear** kernel
3. Load **Sample Dataset**
4. Process & Train
5. View **Overview** → **Feature Importance**

### Workflow 2: Fraud Detection
1. Select **Fraud** domain
2. Choose **RBF** kernel
3. Load **Sample Dataset**
4. Process & Train
5. View **ROC Curve** → **Confusion Matrix**

### Workflow 3: Custom Data
1. Select **Classification** domain
2. Choose **Polynomial** kernel
3. Upload your **CSV file**
4. Review validation warnings
5. Process & Train
6. Explore all result sections

---

## Performance Notes

- Training time: typically < 1 second
- Visualization rendering: < 1 second each
- App load time: ~3-5 seconds
- Best with files under 10MB

---

**Happy Classifying! 🤖**
