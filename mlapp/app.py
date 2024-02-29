import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # Necessary import to use IterativeImputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pickle

# Why "from sklearn.experimental import enable_iterative_imputer" important for IterativeImputer?
# 1. Experimental Status: The IterativeImputer class is marked as experimental, which means it's not considered stable and might have issues or undergo significant changes in future versions of scikit-learn
# 2. Enable: This line is necessary to enable the IterativeImputer class, which allows you to use the IterativeImputer class in your code.
# 3. Explicit Activation: This makes it clear in your code that you are intentionally opting into using this feature, as opposed to accidentally relying on an unstable or experimental feature that might change unexpectedly.
# 4. Encourages Feedback and Testing: By requiring users to explicitly activate experimental features, scikit-learn encourages users to provide feedback, report issues, and participate in testing. This helps improve the quality and reliability of experimental features over time.
# 5. Awareness of Risks: By importing enable_iterative_imputer, users are made aware of the potential risks associated with using IterativeImputer and can make informed decisions about whether it's appropriate for their use case.

# Function to preprocess data
def preprocess_data(X, y, problem):
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['number']).columns
    
    # Define the preprocessing for numerical columns (impute then scale)
    numerical_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer()),
        ('scaler', StandardScaler())
    ])
    
    # Define the preprocessing for categorical columns (impute then one-hot encode)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y

# Function to train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

# Main application function
def main():
    st.title("Machine Learstreaning Application")
    st.write("Welcome to the machine learning application. This app allows you to train and evaluate different machine learning models on your dataset.")
    
    # Initialize data as an empty DataFrame
    data = pd.DataFrame()
    
    data_source = st.sidebar.selectbox("Do you want to upload data or use example data?", ["Upload", "Example"])
    
    if data_source == "Upload":
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx', 'tsv'])
        if uploaded_file is not None:
            if uploaded_file.type == "text/csv":
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                data = pd.read_excel(uploaded_file)
            elif uploaded_file.type == "text/tab-separated-values":
                data = pd.read_csv(uploaded_file, sep='\t')
    else:
        dataset_name = st.sidebar.selectbox("Select a dataset", ["titanic", "tips", "iris", "dots", "anscombe", "attention", "brain_networks", "car_crashes", "exercise", "flights", "fmri", "mpg", "planets", ])
        data = sns.load_dataset(dataset_name)

    if not data.empty:
        st.write("Data Head:", data.head())
        st.write("Data Shape:", data.shape)
        st.write("Data Description:", data.describe())
        st.write("Data Info:", data.info())
        st.write("Column Names:", data.columns.tolist())
        
        # Select features and target
        features = st.multiselect("Select features columns", data.columns.tolist())
        target = st.selectbox("Select target column", data.columns.tolist())
        problem_type = st.selectbox("Problem Type", ["Classification", "Regression"])
        
        if features and target and problem_type:
            X = data[features]
            y = data[target]
            
            st.write(f"You have selected a {problem_type} problem.")
            
            # Button to start analysis
            if st.button("Run Analysis"):
                # Pre-process data
                X_processed, y_processed = preprocess_data(X, y, problem_type)
                
                # Train-test split
                test_size = st.slider("Select test split size", 0.1, 0.5, 0.2)
                X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=test_size, random_state=42)
                
                # Model selection based on problem type
                model_options = ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVM'] if problem_type == 'Regression' else ['Decision Tree', 'Random Forest', 'SVM']
                selected_model = st.sidebar.selectbox("Select model", model_options)
                
                # Initialize model
                if selected_model == 'Linear Regression':
                    model = LinearRegression()
                elif selected_model == 'Decision Tree':
                    model = DecisionTreeRegressor() if problem_type == 'Regression' else DecisionTreeClassifier()
                elif selected_model == 'Random Forest':
                    model = RandomForestRegressor() if problem_type == 'Regression' else RandomForestClassifier()
                elif selected_model == 'SVM':
                    model = SVR() if problem_type == 'Regression' else SVC()
                    
                # Train and evaluate model
                predictions = train_and_evaluate(X_train, X_test, y_train, y_test, model)
                
                st.write("Model training and evaluation complete. Implement specific metrics display here.")
                
                # Download model, make predictions, and show results
                # Further implementation needed based on application requirements.
                
if __name__ == "__main__":
    main()
