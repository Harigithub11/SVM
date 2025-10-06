"""
Script to generate sample datasets for Classic SVM application
Includes realistic correlations as per 2024 best practices
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Set random seed for reproducibility
np.random.seed(42)

def create_medical_dataset(n_samples=500):
    """
    Generate realistic medical dataset with correlated features
    """
    # Generate medical data with realistic correlations (2024 best practice)
    age = np.random.randint(20, 81, n_samples)

    # Blood pressure correlates with age (more realistic)
    blood_pressure = 80 + (age - 20) * 0.8 + np.random.normal(0, 10, n_samples)
    blood_pressure = np.clip(blood_pressure, 80, 200).astype(int)

    # Cholesterol also correlates with age
    cholesterol = 150 + (age - 20) * 1.2 + np.random.normal(0, 15, n_samples)
    cholesterol = np.clip(cholesterol, 150, 300).astype(int)

    # BMI has weak correlation with age
    bmi = 22 + (age - 50) * 0.05 + np.random.normal(0, 3, n_samples)
    bmi = np.clip(bmi, 15, 40).round(1)

    # Heart rate (slight inverse correlation with age)
    heart_rate = 75 - (age - 50) * 0.1 + np.random.normal(0, 8, n_samples)
    heart_rate = np.clip(heart_rate, 50, 120).astype(int)

    # Glucose level (correlates with BMI and age)
    glucose = 70 + (bmi - 22) * 2 + (age - 50) * 0.3 + np.random.normal(0, 10, n_samples)
    glucose = np.clip(glucose, 60, 200).astype(int)

    # Exercise hours (inverse correlation with age and BMI)
    exercise_hours = 5 - (age - 50) * 0.03 - (bmi - 22) * 0.1 + np.random.normal(0, 1, n_samples)
    exercise_hours = np.clip(exercise_hours, 0, 10).round(1)

    # Diagnosis based on multiple factors
    risk_score = (
        (blood_pressure - 120) * 0.02 +
        (cholesterol - 200) * 0.01 +
        (bmi - 25) * 0.15 +
        (glucose - 100) * 0.02 -
        exercise_hours * 0.3 +
        np.random.normal(0, 1, n_samples)
    )

    # 0: Healthy, 1: At Risk, 2: Disease
    diagnosis = np.zeros(n_samples, dtype=int)
    diagnosis[risk_score > 1] = 1
    diagnosis[risk_score > 3] = 2

    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Blood_Pressure': blood_pressure,
        'Cholesterol': cholesterol,
        'BMI': bmi,
        'Heart_Rate': heart_rate,
        'Glucose_Level': glucose,
        'Exercise_Hours_Per_Week': exercise_hours,
        'Diagnosis': diagnosis
    })

    return df

def create_fraud_dataset(n_samples=600):
    """
    Generate fraud detection dataset
    """
    # Transaction amounts (log-normal distribution)
    transaction_amount = np.random.lognormal(5, 1.5, n_samples)
    transaction_amount = np.clip(transaction_amount, 10, 10000).round(2)

    # Transaction hour (0-23)
    transaction_hour = np.random.randint(0, 24, n_samples)

    # Days since last transaction
    days_since_last = np.random.exponential(3, n_samples)
    days_since_last = np.clip(days_since_last, 0, 30).round(1)

    # Number of transactions in last month
    transactions_last_month = np.random.poisson(15, n_samples)
    transactions_last_month = np.clip(transactions_last_month, 0, 100)

    # Account age in months
    account_age = np.random.exponential(24, n_samples)
    account_age = np.clip(account_age, 1, 120).astype(int)

    # Average transaction amount
    avg_transaction = transaction_amount * np.random.uniform(0.8, 1.2, n_samples)
    avg_transaction = avg_transaction.round(2)

    # Location (categorical)
    locations = ['Domestic', 'International']
    location = np.random.choice(locations, n_samples, p=[0.8, 0.2])

    # Fraud determination based on patterns
    fraud_score = (
        (transaction_amount > 1000).astype(int) * 2 +
        (transaction_hour < 6).astype(int) * 1.5 +
        (days_since_last > 20).astype(int) * 1 +
        (account_age < 6).astype(int) * 2 +
        (location == 'International').astype(int) * 1.5 +
        np.random.uniform(0, 1, n_samples)
    )

    # 0: Legitimate, 1: Suspicious, 2: Fraud
    fraud_label = np.zeros(n_samples, dtype=int)
    fraud_label[fraud_score > 3] = 1
    fraud_label[fraud_score > 5] = 2

    # Create DataFrame
    df = pd.DataFrame({
        'Transaction_Amount': transaction_amount,
        'Transaction_Hour': transaction_hour,
        'Days_Since_Last_Transaction': days_since_last,
        'Transactions_Last_Month': transactions_last_month,
        'Account_Age_Months': account_age,
        'Average_Transaction_Amount': avg_transaction,
        'Location': location,
        'Fraud_Label': fraud_label
    })

    return df

def create_classification_dataset(n_samples=350):
    """
    Generate general classification dataset with make_classification
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.0,         # Controls class separability
        flip_y=0.05,          # Label noise for realism
        weights=[0.33, 0.33, 0.34],
        random_state=42        # CRITICAL: For reproducibility
    )

    # Create feature names
    feature_names = [f'Feature_{i+1}' for i in range(8)]

    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['Class'] = y

    return df

if __name__ == "__main__":
    # Create datasets
    print("Creating medical dataset...")
    medical_df = create_medical_dataset(500)
    medical_df.to_csv('medical_sample.csv', index=False)
    print(f"[OK] Medical dataset created: {medical_df.shape}")
    print(f"   Class distribution:\n{medical_df['Diagnosis'].value_counts()}\n")

    print("Creating fraud dataset...")
    fraud_df = create_fraud_dataset(600)
    fraud_df.to_csv('fraud_sample.csv', index=False)
    print(f"[OK] Fraud dataset created: {fraud_df.shape}")
    print(f"   Class distribution:\n{fraud_df['Fraud_Label'].value_counts()}\n")

    print("Creating classification dataset...")
    classification_df = create_classification_dataset(350)
    classification_df.to_csv('classification_sample.csv', index=False)
    print(f"[OK] Classification dataset created: {classification_df.shape}")
    print(f"   Class distribution:\n{classification_df['Class'].value_counts()}\n")

    print("[SUCCESS] All sample datasets created successfully!")
