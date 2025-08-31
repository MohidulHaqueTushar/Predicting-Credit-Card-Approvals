from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# ---- Path handling anchored to this file ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # .../Predicting-Credit-Card-Approvals
def abs_path(relative: str) -> str:
    """Return an absolute path inside the project, regardless of the current working directory."""
    return str((PROJECT_ROOT / relative).resolve())

def load_data(data_name: str) -> pd.DataFrame:
    """Loads data from the specified project-relative path."""
    path = Path(abs_path(data_name))
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found at {path}\n"
            f"Expected under {PROJECT_ROOT}. Check your folder layout: {PROJECT_ROOT}/Data/cc_approvals.data"
        )
    return pd.read_csv(path, header=None)

def preprocess_data(df):
    df = df.replace("?", np.nan)
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].value_counts().index[0])
        else:
            df[col] = df[col].fillna(df[col].mean())
    df = pd.get_dummies(df, drop_first=True)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values  # 1D target
    return X, y

def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    scaler = StandardScaler()
    rescaledX_train = scaler.fit_transform(X_train)
    rescaledX_test = scaler.transform(X_test)
    logreg = LogisticRegression()
    param_grid = dict(tol=[0.01, 0.001, 0.0001], max_iter=[100, 150, 200])
    grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)
    grid_model_result = grid_model.fit(rescaledX_train, y_train.ravel())
    print(f"Best training score: {grid_model_result.best_score_:.4f} using {grid_model_result.best_params_}")
    best_model = grid_model_result.best_estimator_
    print(f"Accuracy of logistic regression classifier: {best_model.score(rescaledX_test, y_test):.4f}")

def main():
    df = load_data("Data/cc_approvals.data")
    X, y = preprocess_data(df)
    train_and_evaluate_model(X, y)

if __name__ == "__main__":
    main()
