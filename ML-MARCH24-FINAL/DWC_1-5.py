import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("DWCData-Plant1-5.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['plant_no', 'date']).reset_index(drop=True)

# =========================
# CUSTOM IMPUTATION
# =========================
def custom_impute(group):
    group = group.sort_values('date')
    for col in group.columns:
        if group[col].dtype != 'object':
            for i in range(len(group)):
                if pd.isna(group.iloc[i][col]):
                    if i > 0 and i < len(group)-1:
                        prev_val = group.iloc[i-1][col]
                        next_val = group.iloc[i+1][col]
                        if not pd.isna(prev_val) and not pd.isna(next_val):
                            group.iloc[i, group.columns.get_loc(col)] = (prev_val + next_val) / 2
    return group

df = df.groupby('plant_no').apply(custom_impute).reset_index(drop=True)

# =========================
# FEATURES & TARGETS
# =========================
features = ['ave_ph', 'ave_do', 'ave_tds', 'ave_temp', 'ave_humidity']
targets = ['height', 'length', 'weight', 'leaves', 'branches']

# =========================
# TRAIN TEST SPLIT
# =========================
train_df = df[df['plant_no'].isin([1,2,3])]
test_df = df[df['plant_no'].isin([4,5])]

X_train = train_df[features]
X_test = test_df[features]

# =========================
# SCALER FOR SVR
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# METRICS FUNCTION
# =========================
def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse

# =========================
# HYPERPARAMETER GRIDS
# =========================
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10]
}

param_grid_svr = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# =========================
# TRAIN & EVALUATE PER TARGET
# =========================
all_results = {}

for target in targets:
    print(f"\n=== TARGET: {target} ===")

    y_train = train_df[target]
    y_test = test_df[target]

    # ---- BASE MODELS ----
    base_models = {
        "MLR": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "SVR": SVR()
    }

    base_results = {}
    tuned_models = {}

    for name, model in base_models.items():
        # choose scaled X for SVR
        Xtr, Xte = (X_train_scaled, X_test_scaled) if name=="SVR" else (X_train, X_test)

        model.fit(Xtr, y_train)
        y_pred_train = model.predict(Xtr)
        y_pred_test = model.predict(Xte)
        r2_train, mae_train, rmse_train = evaluate(y_train, y_pred_train)
        r2_test, mae_test, rmse_test = evaluate(y_test, y_pred_test)

        base_results[name] = {
            "train": (r2_train, mae_train, rmse_train),
            "test": (r2_test, mae_test, rmse_test)
        }

    # ---- HYPERPARAMETER TUNING ----
    # Random Forest
    rf = RandomForestRegressor(random_state=42)
    grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='r2', verbose=0)
    grid_rf.fit(X_train, y_train)
    tuned_models["RandomForest"] = grid_rf.best_estimator_

    # SVR
    svr = SVR()
    grid_svr = GridSearchCV(svr, param_grid_svr, cv=3, scoring='r2', verbose=0)
    grid_svr.fit(X_train_scaled, y_train)
    tuned_models["SVR"] = grid_svr.best_estimator_

    # MLR (no tuning)
    tuned_models["MLR"] = base_models["MLR"]

    # ---- EVALUATE TUNED MODELS ----
    tuned_results = {}
    for name, model in tuned_models.items():
        Xtr, Xte = (X_train_scaled, X_test_scaled) if name=="SVR" else (X_train, X_test)
        y_pred_train = model.predict(Xtr)
        y_pred_test = model.predict(Xte)
        r2_train, mae_train, rmse_train = evaluate(y_train, y_pred_train)
        r2_test, mae_test, rmse_test = evaluate(y_test, y_pred_test)
        tuned_results[name] = {
            "train": (r2_train, mae_train, rmse_train),
            "test": (r2_test, mae_test, rmse_test)
        }

    # ---- PRINT RESULTS ----
    for name in base_models.keys():
        print(f"\nModel: {name}")
        print(" BASE MODEL - Train: R2={:.3f}, MAE={:.3f}, RMSE={:.3f}".format(*base_results[name]["train"]))
        print(" BASE MODEL - Test : R2={:.3f}, MAE={:.3f}, RMSE={:.3f}".format(*base_results[name]["test"]))
        print(" TUNED MODEL - Train: R2={:.3f}, MAE={:.3f}, RMSE={:.3f}".format(*tuned_results[name]["train"]))
        print(" TUNED MODEL - Test : R2={:.3f}, MAE={:.3f}, RMSE={:.3f}".format(*tuned_results[name]["test"]))

    # SAVE TUNED MODELS PER TARGET
    for name, model in tuned_models.items():
        joblib.dump(model, f"DWC_{target}_{name}_model.joblib")

    all_results[target] = tuned_results

# =========================
# SAVE SCALER
# =========================
joblib.dump(scaler, "DWC_scaler.joblib")

# =========================
# VISUALIZATIONS
# Correlation heatmap
# =========================
plt.figure(figsize=(10,8))
sns.heatmap(df[features + targets].corr(), annot=True, fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# =========================
# BAR GRAPH OF R2 PER TARGET
# =========================
for target in targets:
    r2_vals = [all_results[target][m]["test"][0] for m in ["MLR","RandomForest","SVR"]]
    plt.figure()
    bars = plt.bar(["MLR","RF","SVR"], r2_vals)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval,3),
                 ha='center', va='bottom')
    plt.title(f"{target} - R2 Test Comparison")
    plt.ylabel("R2 Score")
    plt.ylim(0,1)
    plt.show()