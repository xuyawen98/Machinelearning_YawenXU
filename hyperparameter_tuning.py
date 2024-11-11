# hyperparameter_tuning.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    df = pd.read_excel(file_path)
    X = df.iloc[:, 8:]  # Features
    y = df.iloc[:, 5]  # Labels
    return X, y


def preprocess_data(X, y):
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Data augmentation and cross-validation
    smote = SMOTE()
    X, y = smote.fit_resample(X, y)
    return X, y


def train_model(X, y):
    # Split data into training and testing sets
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    model = RandomForestClassifier(oob_score=True, random_state=42)
    model.fit(train_x, train_y)
    return model, train_x, test_x, train_y, test_y


def grid_search_random_forest(train_x, train_y):
    param_test1 = {'n_estimators': range(10, 201, 10)}
    param_test2 = {'max_depth': range(3, 30, 1), 'min_samples_split': range(10, 201, 10)}
    param_test3 = {'min_samples_split': range(10, 201, 10), 'min_samples_leaf': range(5, 201, 5)}
    param_test4 = {'max_features': range(3, 159, 3)}

    # Grid search for each parameter
    gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=20,
                                                             min_samples_leaf=10,
                                                             max_depth=5,
                                                             max_features='sqrt',
                                                             random_state=42),
                            param_grid=param_test1, scoring='roc_auc', cv=10)
    gsearch1.fit(train_x, train_y)

    gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=150,
                                                             min_samples_leaf=10,
                                                             max_features='sqrt',
                                                             oob_score=True,
                                                             random_state=42),
                            param_grid=param_test2, scoring='roc_auc', cv=10)
    gsearch2.fit(train_x, train_y)

    gsearch3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=150, max_depth=18,
                                                             max_features='sqrt',
                                                             oob_score=True,
                                                             random_state=42),
                            param_grid=param_test3, scoring='roc_auc', cv=10)
    gsearch3.fit(train_x, train_y)

    gsearch4 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=150, max_depth=18,
                                                             min_samples_split=5,
                                                             min_samples_leaf=10,
                                                             max_features='sqrt',
                                                             oob_score=True,
                                                             random_state=10),
                            param_grid=param_test4, scoring='roc_auc', cv=10)
    gsearch4.fit(train_x, train_y)

    # Print best parameters
    print('Best parameters for gsearch1:', gsearch1.best_params_)
    print('Best parameters for gsearch2:', gsearch2.best_params_)
    print('Best parameters for gsearch3:', gsearch3.best_params_)
    print('Best parameters for gsearch4:', gsearch4.best_params_)

    return gsearch1, gsearch2, gsearch3, gsearch4


def main():
    file_path = '/Users/xuyawen/PycharmProjects/fNIRS/myenv/0109调换顺序new.xlsx'
    X, y = load_data(file_path)
    X, y = preprocess_data(X, y)
    model, train_x, test_x, train_y, test_y = train_model(X, y)
    grid_search_random_forest(train_x, train_y)


if __name__ == "__main__":
    main()