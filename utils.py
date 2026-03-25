class AutoScienceHelper:
    """
    A data science tool to simplify common tasks.

    This class provides utility functions to assist with data manipulation and analysis.
    """

import pandas as pd
import numpy as np

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")

def clean_data(data):
    try:
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        return data
    except Exception as e:
        print(f"Error cleaning data: {e}")

def main():
    file_path = 'data.csv'
    data = load_data(file_path)
    if data is not None:
        cleaned_data = clean_data(data)
        print(cleaned_data.head())

if __name__ == "__main__":
    main()