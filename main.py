import pandas as pd
import numpy as np
import kagglehub
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import LabelEncoder

def download_and_load_data():
    """
    Downloads the customer churn dataset from Kaggle and loads it into a Pandas DataFrame.
    """
    path = kagglehub.dataset_download("muhammadshahidazeem/customer-churn-dataset")

    # The actual filename is 'customer_churn_dataset-training-master.csv'
    file_path = os.path.join(path, "customer_churn_dataset-training-master.csv")

    # Load the dataset into a DataFrame
    df = pd.read_csv(file_path)
    return df

def encode_categorical_columns(df):
    """
    Encodes categorical columns ('Gender', 'Subscription Type', 'Contract Length') into numerical values.
    
    Parameters:
    - df (DataFrame): Input DataFrame containing the customer data.
    
    Returns:
    - df (DataFrame): DataFrame with encoded categorical columns.
    """
    encoder = LabelEncoder()

    # Encode the categorical columns and print category-numerical mappings
    for column in ['Gender', 'Subscription Type', 'Contract Length']:
        df[column] = encoder.fit_transform(df[column])

        # Print category-numerical pairs for each encoded column
        print(f"\nCategory-Numerical Pairs for {column}:")
        for category, numerical in zip(encoder.classes_, encoder.transform(encoder.classes_)):
            print(f"{category}: {numerical}")

    return df

def prepare_features_and_labels(df):
    """
    Prepares the feature matrix X and the target vector y for training and testing.
    
    Parameters:
    - df (DataFrame): DataFrame with the dataset.
    
    Returns:
    - X (ndarray): Feature matrix for model training.
    - y (ndarray): Target vector for model training.
    """
    selected_features = ['Age', 'Gender', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction', 'Churn']
    X = df[selected_features]
    y = df['Churn']
    return X, y

def scale_features(X):
    """
    Scales the feature matrix using StandardScaler.
    
    Parameters:
    - X (ndarray): Feature matrix.
    
    Returns:
    - X_scaled (ndarray): Scaled feature matrix.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model on the given training data.
    
    Parameters:
    - X_train (ndarray): Training feature matrix.
    - y_train (ndarray): Training target vector.
    
    Returns:
    - model (LogisticRegression): Trained logistic regression model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("...Training Logistic Regressor")
    return model


def save_model(model, filename='logistic_regression_model.pkl'):
    """
    Saves the trained model to a pickle file.
    
    Parameters:
    - model (sklearn model): Trained model to be saved.
    - filename (str): Filename where the model will be saved (default is 'logistic_regression_model.pkl').
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved as {filename}")


def main():
    """
    Main function to load data, process it, train a model, and save the trained model.
    """
    # Step 1: Download and load dataset
    df = download_and_load_data()

    # Step 2: Encode categorical columns
    df = encode_categorical_columns(df)

    # Step 3: Prepare features and labels
    X, y = prepare_features_and_labels(df)

    # Step 4: Scale features
    X_scaled = scale_features(X)

    # Step 5: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Further split the training set into train and validation sets (70% train, 30% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Step 6: Train the Logistic Regression model
    model = train_logistic_regression(X_train, y_train)

    # Step 7: Save the trained model
    save_model(model)


if __name__ == "__main__":
    main()



