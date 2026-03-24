import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, PowerTransformer
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
# OUTLIER REMOVAL (Z-SCORE)
# -----------------------------
all_cols = ['ave_ph', 'ave_do', 'ave_tds', 'ave_temp', 'ave_humidity',
            'height', 'length', 'weight', 'leaves', 'branches']
z_scores = np.abs(zscore(df[all_cols]))
df = df[(z_scores < 3).all(axis=1)].reset_index(drop=True)

# -----------------------------
# FEATURES & TARGETS
# -----------------------------
features = ['ave_ph', 'ave_do', 'ave_tds', 'ave_temp', 'ave_humidity']
targets = ['height', 'length', 'weight', 'leaves', 'branches']
X = df[features]
y = df[targets]

# -----------------------------
# TARGET TRANSFORMATION
# -----------------------------
pt = PowerTransformer(method='yeo-johnson')  # works with zero or negative
y_transformed = pd.DataFrame(pt.fit_transform(y), columns=targets)

# -----------------------------
# HEATMAP OF CORRELATIONS
# -----------------------------
plt.figure(figsize=(10, 8))
corr_matrix = pd.concat([X, y], axis=1).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Feature and Target Correlation Heatmap")
plt.show()

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
train_indices, test_indices = [], []
for date, group in df.groupby('date'):
    train_indices.extend(group[group['plant_no'].isin([1,2,3])].index)
    test_indices.extend(group[group['plant_no'].isin([4,5])].index)

X_train = X.loc[train_indices]
X_test = X.loc[test_indices]
y_train = y_transformed.loc[train_indices]
y_test = y_transformed.loc[test_indices]

# -----------------------------
# SCALING FEATURES
# -----------------------------
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

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
    for plant in [4,5]:
        plant_indices = df[df['plant_no']==plant].index.intersection(test_indices)
        idx_map = [i for i, idx in enumerate(test_indices) if idx in plant_indices]
        y_plant_true = y_test.iloc[idx_map]   # <-- use iloc
        y_plant_pred = y_test_pred[idx_map]   # <-- use iloc
        print(f"\nPlant {plant}:")
        base_results[name][f"Plant {plant}"] = {}
        for i, target in enumerate(targets):
            r2 = r2_score(y_plant_true.iloc[:, i], y_plant_pred[:, i])
            mae = mean_absolute_error(y_plant_true.iloc[:, i], y_plant_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_plant_true.iloc[:, i], y_plant_pred[:, i]))
            base_results[name][f"Plant {plant}"][target] = {"R2": r2, "MAE": mae, "RMSE": rmse}
            print(f"{target}: R2={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")

            # Scatter plot
            plt.figure()
            plt.scatter(y_plant_true.iloc[:, i], y_plant_pred[:, i], alpha=0.7)
            min_val = min(y_plant_true.iloc[:, i].min(), y_plant_pred[:, i].min())
            max_val = max(y_plant_true.iloc[:, i].max(), y_plant_pred[:, i].max())
            plt.plot([min_val, max_val], [min_val, max_val], '--r')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f"{name} - Plant {plant} - {target}\nR2={r2:.3f}")
            plt.show()

# -----------------------------
# HYPERPARAMETER TUNING (Random Forest)
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

tuned_models = {"Random Forest (Tuned)": rf_best}
tuned_results = {}

# Evaluation for tuned models
for name, model in tuned_models.items():
    y_test_pred = model.predict(X_test)
    tuned_results[name] = {}
    for plant in [4,5]:
        plant_indices = df[df['plant_no']==plant].index.intersection(test_indices)
        idx_map = [i for i, idx in enumerate(test_indices) if idx in plant_indices]
        y_plant_true = y_test.iloc[idx_map]   # <-- iloc
        y_plant_pred = y_test_pred[idx_map]   # <-- iloc
        tuned_results[name][f"Plant {plant}"] = {}
        print(f"\n{name} - Plant {plant} Metrics:")
        for i, target in enumerate(targets):
            r2 = r2_score(y_plant_true.iloc[:, i], y_plant_pred[:, i])
            mae = mean_absolute_error(y_plant_true.iloc[:, i], y_plant_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_plant_true.iloc[:, i], y_plant_pred[:, i]))
            tuned_results[name][f"Plant {plant}"][target] = {"R2": r2, "MAE": mae, "RMSE": rmse}
            print(f"{target}: R2={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")

# -----------------------------
# BAR PLOTS: Base vs Tuned (Per Plant)
# -----------------------------
metrics = ["R2","MAE","RMSE"]
for plant in [4,5]:
    for target in targets:
        plt.figure()
        values = []
        labels = []
        for name in base_models:
            base_val = base_results[name][f"Plant {plant}"][target]["R2"]
            values.append(base_val)
            labels.append(name+" (Base)")
            tuned_name = name + " (Tuned)"
            if tuned_name in tuned_results:
                tuned_val = tuned_results[tuned_name][f"Plant {plant}"][target]["R2"]
                values.append(tuned_val)
                labels.append(tuned_name)
        bars = plt.bar(labels, values, color=['skyblue','orange','skyblue','orange','skyblue','orange'])
        plt.title(f"R2 Comparison - {target} (Plant {plant})")
        plt.xticks(rotation=45)
        for bar in bars:
            plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                     f"{bar.get_height():.2f}", ha='center', va='bottom')
        plt.show()
