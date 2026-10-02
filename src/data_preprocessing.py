# data_preprocessing.py

import pandas as pd
import numpy as np


def load_data(file_path):
    """
    Load dataset from CSV file
    """
    df = pd.read_csv(file_path)
    print("Data Loaded Successfully")
    print("Shape:", df.shape)
    return df


def handle_missing_values(df):
    """
    Handle missing values using basic strategies
    """
    print("\nHandling Missing Values...")

    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def remove_duplicates(df):
    """
    Remove duplicate rows
    """
    print("\nRemoving Duplicates...")
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Removed {before - after} duplicate rows")
    return df


def handle_outliers(df):
    """
    Remove outliers using IQR method
    """
    print("\nHandling Outliers...")

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


def normalize_data(df):
    """
    Normalize numerical columns
    """
    print("\nNormalizing Data...")

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    return df


def preprocess_pipeline(file_path):
    """
    Complete preprocessing pipeline
    """
    df = load_data(file_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df = normalize_data(df)

    print("\nPreprocessing Completed!")
    print("Final Shape:", df.shape)

    return df


if __name__ == "__main__":
    # Example usage
    file_path = "data/sample_data.csv"  # Change this path as needed
    processed_data = preprocess_pipeline(file_path)

    # Save cleaned data
    processed_data.to_csv("data/processed_data.csv", index=False)
    print("\nProcessed data saved successfully!")
