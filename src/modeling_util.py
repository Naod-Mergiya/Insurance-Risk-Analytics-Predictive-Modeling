import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import shap


def handle_missing(df, strategy='mean'):
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            if strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().sum() > 0:
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode[0])
            else:
                df[col] = df[col].fillna('missing')
    return df

def encode_categoricals(df):
    return pd.get_dummies(df, drop_first=True)

def feature_engineering(df):
    df = df.copy()
    # Example: vehicle age
    if 'RegistrationYear' in df.columns and 'TransactionMonth' in df.columns:
        df['TransactionYear'] = pd.to_datetime(df['TransactionMonth']).dt.year
        df['VehicleAge'] = df['TransactionYear'] - df['RegistrationYear']
    return df

def prepare_data(df, target, drop_cols=None, test_size=0.2, random_state=42, regression=True):
    df = handle_missing(df)
    df = feature_engineering(df)

    if drop_cols:
        safe_drop_cols = [col for col in drop_cols if col != target]
        if safe_drop_cols:
            df = df.drop(columns=safe_drop_cols)
    X = df.drop(columns=[target])
    y = df[target]
    X = encode_categoricals(X)

    X = X.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, regression=True, random_state=42):
    if regression:
        model = RandomForestRegressor(random_state=random_state)
    else:
        model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def filter_flat_numeric_columns(df):
    import numpy as np
    import pandas as pd
    def is_flat_numeric_col(s):
        if not isinstance(s, pd.Series):
            return False
        for v in s:
            if isinstance(v, (list, tuple, np.ndarray, pd.DataFrame, pd.Series)) and not pd.isna(v):
                return False
        return True
    flat_cols = [col for col in df.columns if is_flat_numeric_col(df[col])]
    df = df[flat_cols]
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.select_dtypes(include=[np.number]).astype('float64')
    return df

def train_xgboost(X_train, y_train, regression=True, random_state=42):
    from xgboost import XGBRegressor, XGBClassifier
    X_train = filter_flat_numeric_columns(X_train)
    if regression:
        model = XGBRegressor(random_state=random_state, verbosity=0)
    else:
        model = XGBClassifier(random_state=random_state, use_label_encoder=False, verbosity=0)
    model.fit(X_train, y_train)
    return model

# Model Evaluation

def regression_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'R2': r2}

def classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}


def get_feature_importance(model, X_train):
    if hasattr(model, 'feature_importances_'):
        return pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    elif hasattr(model, 'coef_'):
        return pd.Series(model.coef_, index=X_train.columns).sort_values(ascending=False)
    else:
        return None

def shap_summary_plot(model, X_train, max_display=10):
    # Coerce any lingering object columns to numeric
    X_numeric = X_train.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")

    explainer = shap.Explainer(model, X_numeric)
    # Disable expensive additivity check that can fail due to numeric coercion
    shap_values = explainer(X_numeric, check_additivity=False)
    shap.summary_plot(shap_values, X_numeric, max_display=max_display)