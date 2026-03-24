import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("AeroData.csv")

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
# CORRELATION HEATMAP
# -----------------------------
plt.figure(figsize=(10,8))
corr = df.corr(numeric_only=True)

plt.imshow(corr, aspect='auto')
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                 ha='center', va='center', fontsize=8)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# -----------------------------
# FEATURES & TARGETS
# -----------------------------
features = ['ave_ph', 'ave_do', 'ave_tds', 'ave_temp', 'ave_humidity']
targets = ['height', 'length', 'weight', 'leaves', 'branches']

X = df[features]
y = df[targets]

# -----------------------------
# TRAIN-TEST SPLIT PER DAY
# -----------------------------
train_indices, test_indices = [], []

for date, group in df.groupby('date'):
    indices = list(group.index)
    random.shuffle(indices)
    train_indices.extend(indices[:4])
    test_indices.extend(indices[4:6])

X_train = X.loc[train_indices]
X_test = X.loc[test_indices]
y_train = y.loc[train_indices]
y_test = y.loc[test_indices]

# -----------------------------
# VISUALIZE SPLIT
# -----------------------------
df['set'] = 'Train'
df.loc[test_indices, 'set'] = 'Test'

plt.figure(figsize=(12,6))
for label in ['Train', 'Test']:
    subset = df[df['set'] == label]
    plt.scatter(subset['date'], subset['plant_no'], label=label)

plt.title("Train-Test Split per Day")
plt.xlabel("Date")
plt.ylabel("Plant No")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# MODELS
# -----------------------------
models = {
    "MLR": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": MultiOutputRegressor(SVR())
}

results = {model: {} for model in models}

# -----------------------------
# TRAIN, STORE, AND PRINT RESULTS
# -----------------------------
for name, model in models.items():

    print(f"\n===== {name} =====")

    if name == "SVR":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    for i, target in enumerate(targets):
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))

        results[name][target] = {
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse
        }

        print(f"\nTarget: {target}")
        print(f"R2   : {r2:.4f}")
        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")

# -----------------------------
# MODEL COMPARISON
# -----------------------------
for metric in ["R2", "MAE", "RMSE"]:
    for target in targets:
        plt.figure()

        values = [results[m][target][metric] for m in models]
        bars = plt.bar(models.keys(), values)

        plt.title(f"{metric} Comparison - {target}")

        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height(),
                     f"{bar.get_height():.2f}",
                     ha='center')

        plt.show()

# -----------------------------
# BEST MODEL PER TARGET
# -----------------------------
best_models = {}
for target in targets:
    best = max(models.keys(), key=lambda m: results[m][target]["R2"])
    best_models[target] = results[best][target]["R2"]

plt.figure()
bars = plt.bar(best_models.keys(), best_models.values())

plt.title("Best Model per Target (R2)")

for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height(),
             f"{bar.get_height():.2f}",
             ha='center')

plt.show()

# -----------------------------
# FEATURE IMPORTANCE (RF)
# -----------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_

plt.figure()
bars = plt.bar(features, importances)

plt.title("Feature Importance (Random Forest)")

for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height(),
             f"{bar.get_height():.3f}",
             ha='center')

plt.xticks(rotation=45)
plt.show()

# -----------------------------
# ACTUAL VS PREDICTED (SCATTER)
# -----------------------------
for name, model in models.items():

    if name == "SVR":
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)

    for i, target in enumerate(targets):

        actual = y_test.iloc[:, i].values
        predicted = y_pred[:, i]

        plt.figure()
        plt.scatter(actual, predicted)

        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())

        plt.plot([min_val, max_val],
                 [min_val, max_val],
                 linestyle='--')

        r2 = r2_score(actual, predicted)

        plt.title(f"{name} - {target}\nR2 = {r2:.3f}")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")

        plt.show()