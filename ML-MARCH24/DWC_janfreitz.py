import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

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
# FEATURES & TARGETS
# -----------------------------
features = ['ave_ph', 'ave_do', 'ave_tds', 'ave_temp', 'ave_humidity']
targets = ['height', 'length', 'weight', 'leaves', 'branches']
X = df[features]
y = df[targets]

# -----------------------------
# HEATMAP OF CORRELATIONS
# -----------------------------
plt.figure(figsize=(10, 8))
corr_matrix = pd.concat([X, y], axis=1).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Feature and Target Correlation Heatmap")
plt.show()

# -----------------------------
# TRAIN-TEST SPLIT (FIXED)
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

# -----------------------------
# TUNED MODELS (Fixed Trees)
# -----------------------------
tuned_models = {
    "ExtraTrees_500": ExtraTreesRegressor(n_estimators=500, random_state=42),
    "RandomForest_400": RandomForestRegressor(n_estimators=400, random_state=42)
}

# -----------------------------
# TRAIN & EVALUATE ALL MODELS
# -----------------------------
all_models = {**base_models, **tuned_models}
results = {}

for name, model in all_models.items():
    if "SVR" in name:
        model.fit(X_train_scaled, y_train)
        y_test_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
    
    results[name] = {}
    
    for plant in [4, 5]:
        plant_indices = df[df['plant_no']==plant].index.intersection(test_indices)
        y_plant_true = y.loc[plant_indices]
        idx_map = [i for i, idx in enumerate(test_indices) if idx in plant_indices]
        y_plant_pred = y_test_pred[idx_map]
        
        results[name][f"Plant {plant}"] = {}
        print(f"\n=== {name} Metrics for Plant {plant} ===")
        
        for i, target in enumerate(targets):
            r2 = r2_score(y_plant_true.iloc[:, i], y_plant_pred[:, i])
            mae = mean_absolute_error(y_plant_true.iloc[:, i], y_plant_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_plant_true.iloc[:, i], y_plant_pred[:, i]))
            
            results[name][f"Plant {plant}"][target] = {"R2": r2, "MAE": mae, "RMSE": rmse}
            print(f"{target}: R2={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")
            
            # Scatter plot per plant
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
# BAR PLOTS: Per Plant, Test Only
# -----------------------------
metrics = ["R2","MAE","RMSE"]
for plant in [4,5]:
    for target in targets:
        for metric in metrics:
            plt.figure()
            values = []
            labels = []
            for name in all_models:
                val = results[name][f"Plant {plant}"][target][metric]
                values.append(val)
                labels.append(name)
            bars = plt.bar(labels, values, color=['skyblue','orange','green','red','purple'])
            plt.title(f"{metric} Comparison - {target} (Plant {plant})")
            plt.xticks(rotation=45)
            for bar in bars:
                plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                         f"{bar.get_height():.2f}", ha='center', va='bottom')
            plt.show()