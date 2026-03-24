import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("Feb6-19_DWCDATA.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date').reset_index(drop=True)

# -----------------------------
# MISSING VALUE IMPUTATION
# -----------------------------
def impute_by_neighbor_mean(df, column):
    df = df.copy()
    for i in range(len(df)):
        if pd.isna(df.loc[i, column]):
            prev_val, next_val = None, None
            for j in range(i-1, -1, -1):
                if not pd.isna(df.loc[j, column]):
                    prev_val = df.loc[j, column]
                    break
            for j in range(i+1, len(df)):
                if not pd.isna(df.loc[j, column]):
                    next_val = df.loc[j, column]
                    break
            if prev_val is not None and next_val is not None:
                df.loc[i, column] = (prev_val + next_val) / 2
    return df

cols_to_impute = ['ave_ph', 'ave_do', 'ave_tds', 'ave_temp', 'ave_humidity']
for col in cols_to_impute:
    df = impute_by_neighbor_mean(df, col)

# -----------------------------
# FEATURES & TARGETS
# -----------------------------
features = ['ave_ph', 'ave_do', 'ave_tds', 'ave_temp', 'ave_humidity']
targets = ['height', 'length', 'weight', 'leaves', 'branches']
X = df[features]
y = df[targets]

# -----------------------------
# TRAIN-TEST SPLIT (FIXED)
# -----------------------------
train_indices, test_indices = [], []
for date, group in df.groupby('date'):
    train_indices.extend(group[group['plant_no'].isin([1,2,3,4])].index)
    test_indices.extend(group[group['plant_no'].isin([5,6])].index)

X_train = X.loc[train_indices]
X_test = X.loc[test_indices]
y_train = y.loc[train_indices]
y_test = y.loc[test_indices]

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# BASE MODELS
# -----------------------------
base_models = {
    "MLR": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "SVR": MultiOutputRegressor(SVR())
}
base_results = {}

print("=== Base Models ===")
for name, model in base_models.items():
    if "SVR" in name:
        model.fit(X_train_scaled, y_train)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    
    base_results[name] = {}
    print(f"\n{name} Performance:")
    for i, target in enumerate(targets):
        r2_train = r2_score(y_train.iloc[:, i], y_train_pred[:, i])
        mae_train = mean_absolute_error(y_train.iloc[:, i], y_train_pred[:, i])
        rmse_train = np.sqrt(mean_squared_error(y_train.iloc[:, i], y_train_pred[:, i]))
        
        r2_test = r2_score(y_test.iloc[:, i], y_test_pred[:, i])
        mae_test = mean_absolute_error(y_test.iloc[:, i], y_test_pred[:, i])
        rmse_test = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i]))
        
        base_results[name][target] = {
            "Train": {"R2": r2_train, "MAE": mae_train, "RMSE": rmse_train},
            "Test": {"R2": r2_test, "MAE": mae_test, "RMSE": rmse_test}
        }
        
        print(f"{target}: Train -> R2={r2_train:.3f}, MAE={mae_train:.3f}, RMSE={rmse_train:.3f} | "
              f"Test -> R2={r2_test:.3f}, MAE={mae_test:.3f}, RMSE={rmse_test:.3f}")

# -----------------------------
# HYPERPARAMETER TUNING
# -----------------------------
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42),
                       rf_param_grid, cv=3, scoring='r2', n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_

svr_param_grid = {
    'estimator__C': [0.1, 1, 10],
    'estimator__epsilon': [0.01, 0.1, 0.2],
    'estimator__kernel': ['linear', 'rbf']
}
svr_grid = GridSearchCV(MultiOutputRegressor(SVR()), svr_param_grid,
                        cv=3, scoring='r2', n_jobs=-1)
svr_grid.fit(X_train_scaled, y_train)
svr_best = svr_grid.best_estimator_

tuned_models = {
    "Random Forest (Tuned)": rf_best,
    "SVR (Tuned)": svr_best
}
tuned_results = {}

# -----------------------------
# EVALUATION + VISUALIZATION
# -----------------------------
for name, model in {**base_models, **tuned_models}.items():
    if "SVR" in name:
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
    else:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    
    results_dict = base_results if name in base_models else tuned_results
    results_dict[name] = {}
    print(f"\n=== {name} Metrics ===")
    for i, target in enumerate(targets):
        r2_train = r2_score(y_train.iloc[:, i], y_train_pred[:, i])
        mae_train = mean_absolute_error(y_train.iloc[:, i], y_train_pred[:, i])
        rmse_train = np.sqrt(mean_squared_error(y_train.iloc[:, i], y_train_pred[:, i]))
        
        r2_test = r2_score(y_test.iloc[:, i], y_test_pred[:, i])
        mae_test = mean_absolute_error(y_test.iloc[:, i], y_test_pred[:, i])
        rmse_test = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i]))
        
        results_dict[name][target] = {
            "Train": {"R2": r2_train, "MAE": mae_train, "RMSE": rmse_train},
            "Test": {"R2": r2_test, "MAE": mae_test, "RMSE": rmse_test}
        }
        
        print(f"{target}: Train -> R2={r2_train:.3f}, MAE={mae_train:.3f}, RMSE={rmse_train:.3f} | "
              f"Test -> R2={r2_test:.3f}, MAE={mae_test:.3f}, RMSE={rmse_test:.3f}")

        # Actual vs Predicted scatter plot
        plt.figure()
        plt.scatter(y_test.iloc[:, i], y_test_pred[:, i], alpha=0.7, label='Data points')
        min_val = min(y_test.iloc[:, i].min(), y_test_pred[:, i].min())
        max_val = max(y_test.iloc[:, i].max(), y_test_pred[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], '--r', label='Ideal')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f"{name} - {target}\nR2={r2_test:.3f}")
        plt.legend()
        plt.show()

# -----------------------------
# BAR PLOTS: Base vs Tuned Metrics (Test Only)
# -----------------------------
metrics = ["R2","MAE","RMSE"]
for target in targets:
    for metric in metrics:
        plt.figure()
        values = []
        labels = []
        for name in base_models:
            base_val = base_results[name][target]["Test"][metric]
            values.append(base_val)
            labels.append(name+" (Base)")
            tuned_name = name + " (Tuned)"
            if tuned_name in tuned_results:
                tuned_val = tuned_results[tuned_name][target]["Test"][metric]
                values.append(tuned_val)
                labels.append(tuned_name)
        bars = plt.bar(labels, values, color=['skyblue','orange','skyblue','orange','skyblue','orange'])
        plt.title(f"{metric} Comparison - {target}")
        plt.xticks(rotation=45)
        for bar in bars:
            plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                     f"{bar.get_height():.2f}", ha='center', va='bottom')
        plt.show()