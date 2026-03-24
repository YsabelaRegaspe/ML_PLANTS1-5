import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.stats import zscore

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("DWCData-Plant1-5.csv")
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
# FEATURE ENGINEERING
# -----------------------------
df['temp_humidity'] = df['ave_temp'] * df['ave_humidity']
df['do_ph_ratio'] = df['ave_do'] / df['ave_ph']
df['height_lag1'] = df.groupby('plant_no')['height'].shift(1)
df['height_lag1'] = df['height_lag1'].fillna(df['height'].mean())

# -----------------------------
# OUTLIER REMOVAL
# -----------------------------
all_cols = ['ave_ph', 'ave_do', 'ave_tds', 'ave_temp', 'ave_humidity',
            'temp_humidity', 'do_ph_ratio', 'height', 'length', 'weight', 'leaves', 'branches', 'height_lag1']
z_scores = np.abs(zscore(df[all_cols]))
df = df[(z_scores < 3).all(axis=1)].reset_index(drop=True)

# -----------------------------
# FEATURES & TARGETS
# -----------------------------
features = ['ave_ph', 'ave_do', 'ave_tds', 'ave_temp', 'ave_humidity',
            'temp_humidity', 'do_ph_ratio', 'height_lag1']
targets = ['height', 'length', 'weight', 'leaves', 'branches']
X = df[features]
y = df[targets]

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
train_indices, test_indices = [], []
for date, group in df.groupby('date'):
    train_indices.extend(group[group['plant_no'].isin([1,2,3])].index)
    test_indices.extend(group[group['plant_no'].isin([4,5])].index)

X_train = X.loc[train_indices]
X_test = X.loc[test_indices]
y_train = y.loc[train_indices]
y_test = y.loc[test_indices]

# -----------------------------
# SCALING
# -----------------------------
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# -----------------------------
# MODELS PER TARGET
# -----------------------------
models = {}
results = {}

for target in targets:
    print(f"\n========== TARGET: {target} ==========")
    
    # Log-transform for height
    if target == 'height':
        y_train_target = np.log1p(y_train[target])
        y_test_target = np.log1p(y_test[target])
    else:
        y_train_target = y_train[target]
        y_test_target = y_test[target]

    # -----------------------------
    # BASE MODELS
    # -----------------------------
    base_models = {
        "MLR": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "SVR": SVR()
    }

    # -----------------------------
    # HYPERPARAMETER TUNING
    # -----------------------------
    print("\n--- Hyperparameter Tuning ---")

    rf_param_grid = {
        'n_estimators': [100, 200, 400],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf_grid = GridSearchCV(RandomForestRegressor(random_state=42),
                           rf_param_grid, cv=3, scoring='r2', n_jobs=-1)
    rf_grid.fit(X_train, y_train_target)
    rf_best = rf_grid.best_estimator_
    print("Best RF Params:", rf_grid.best_params_)

    svr_param_grid = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.2],
        'kernel': ['linear', 'rbf']
    }

    svr_grid = GridSearchCV(SVR(), svr_param_grid,
                            cv=3, scoring='r2', n_jobs=-1)
    svr_grid.fit(X_train_scaled, y_train_target)
    svr_best = svr_grid.best_estimator_
    print("Best SVR Params:", svr_grid.best_params_)

    tuned_models = {
        "RF_Tuned": rf_best,
        "SVR_Tuned": svr_best
    }

    all_models = {**base_models, **tuned_models}
    results[target] = {}

    # -----------------------------
    # TRAIN & EVALUATE
    # -----------------------------
    for name, model in all_models.items():

        if "SVR" in name:
            model.fit(X_train_scaled, y_train_target)
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train_target)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

        # Inverse transform for height
        if target == 'height':
            y_train_pred = np.expm1(y_train_pred)
            y_test_pred = np.expm1(y_test_pred)
            y_train_true = y_train[target]
            y_test_true = y_test[target]
        else:
            y_train_true = y_train_target
            y_test_true = y_test_target

        # Metrics
        r2_train = r2_score(y_train_true, y_train_pred)
        r2_test = r2_score(y_test_true, y_test_pred)
        mae_test = mean_absolute_error(y_test_true, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test_true, y_test_pred))

        print(f"\n{name}")
        print(f"Train R2: {r2_train:.3f}")
        print(f"Test  R2: {r2_test:.3f}, MAE: {mae_test:.3f}, RMSE: {rmse_test:.3f}")

        # -----------------------------
        # PER-PLANT SCATTER
        # -----------------------------
        for plant in [4,5]:
            plant_indices = df[df['plant_no']==plant].index.intersection(test_indices)
            idx_map = [i for i, idx in enumerate(test_indices) if idx in plant_indices]

            y_true_plant = y_test[target].iloc[idx_map]
            if target == 'height':
                y_pred_plant = np.expm1(y_test_pred[idx_map])
            else:
                y_pred_plant = y_test_pred[idx_map]

            plt.figure()
            plt.scatter(y_true_plant, y_pred_plant, alpha=0.7)
            min_val = min(y_true_plant.min(), y_pred_plant.min())
            max_val = max(y_true_plant.max(), y_pred_plant.max())
            plt.plot([min_val, max_val], [min_val, max_val], '--r')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f"{name} - Plant {plant} - {target}")
            plt.show()

# -----------------------------
# HEATMAP
# -----------------------------
plt.figure(figsize=(10,8))
corr_matrix = pd.concat([X, y], axis=1).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature and Target Correlation Heatmap")
plt.show()