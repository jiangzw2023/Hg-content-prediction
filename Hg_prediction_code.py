import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from ngboost import NGBRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import os
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Arial'

# 1. Data Reading and Cleaning
data = pd.read_excel('D:/Compiled_dataset.xlsx', 
                     sheet_name=['Jizhentun', 'Xiahuayuan', 'This study', 'Ediacaran', 'Yuertusi', 'Wayao', 'LSK', 'Tournaisian', 'Duli', 'Dupont GHS', 'Nandong'])
df = pd.concat(data.values(), ignore_index=True)

df = df[df['Hg'].notna()]
Hg_mean, Hg_std = df['Hg'].mean(), df['Hg'].std()
df = df[(df['Hg'] >= Hg_mean - 3*Hg_std) & (df['Hg'] <= Hg_mean + 3*Hg_std)]

# 2. Feature Engineering
df['TOC_TS'] = df['TOC'] * df['TS']
df['Mo_Al'] = df['Mo'] / (df['Al'] + 1e-6)
df['P_Al'] = df['P'] / (df['Al'] + 1e-6)
features = ['TOC', 'TS', 'Mo', 'P', 'Al', 'TOC_TS', 'Mo_Al', 'P_Al']

# 3. Normalization + Polynomial Features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features])
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
Y = df['Hg'].values

# 4. Split Data and Yeo-Johnson Transformation
X_train, X_test, Y_train, Y_test = train_test_split(X_poly, Y, test_size=0.2, random_state=42)
transformer = PowerTransformer(method='yeo-johnson')
Y_train_trans = transformer.fit_transform(Y_train.reshape(-1, 1)).ravel()

# 5. Model Definitions
models = {
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'SVR': SVR(C=10, gamma='scale'),
    'MLP': MLPRegressor(hidden_layer_sizes=(50,), alpha=0.001, max_iter=1000, random_state=42),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, num_leaves=31, random_state=42),
    'CatBoost': cb.CatBoostRegressor(iterations=200, depth=6, verbose=0, random_state=42)
}

# NGBoost defined separately (not used in stacking)
ngboost_model = NGBRegressor(n_estimators=100, verbose=False, random_state=42)

# TabNet data preprocessing
X_tabnet_raw = scaler.fit_transform(df[features]).astype(np.float32)
Y_tabnet_raw = df['Hg'].values

X_train_tab, X_test_tab, Y_train_tab, Y_test_tab = train_test_split(X_tabnet_raw, Y_tabnet_raw, test_size=0.2, random_state=42)
Y_train_tab_trans = transformer.fit_transform(Y_train_tab.reshape(-1, 1)).ravel()

tabnet_model = TabNetRegressor(
    n_d=16, n_a=16, n_steps=5,
    gamma=1.5,
    lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=1e-3),
    verbose=0, seed=42
)
tabnet_model.fit(
    X_train_tab, Y_train_tab_trans.reshape(-1, 1),
    max_epochs=300, patience=30,
    batch_size=32, virtual_batch_size=64
)

# 6. Model Evaluation and Visualization
fig, axs = plt.subplots(3, 3, figsize=(18, 15))
axs = axs.ravel()

metrics = {}
letters = list("ABCDEFGHI")

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, Y_train_trans)
    Y_pred_trans = model.predict(X_test)
    Y_pred = transformer.inverse_transform(Y_pred_trans.reshape(-1, 1)).ravel()
    r2 = r2_score(Y_test, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    mae = mean_absolute_error(Y_test, Y_pred)
    metrics[name] = (r2, rmse, mae)

    axs[i].scatter(Y_test, Y_pred, edgecolors='k', alpha=0.7)
    axs[i].plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
    axs[i].set_title(f"{name}\nR²={r2:.2f} | RMSE={rmse:.2f} | MAE={mae:.2f}", fontsize=12)
    axs[i].set_xlabel("Actual Hg", fontsize=10)
    axs[i].set_ylabel("Predicted Hg", fontsize=10)
    axs[i].text(0.05, 0.92, f'({letters[i]})', transform=axs[i].transAxes,
                fontsize=14, fontweight='bold', ha='left', va='top')
    axs[i].grid(True)
    
# NGBoost Evaluation 
ngboost_model.fit(X_train, Y_train_trans)
Y_pred_trans_ng = ngboost_model.predict(X_test)
Y_pred_ng = transformer.inverse_transform(Y_pred_trans_ng.reshape(-1, 1)).ravel()
r2 = r2_score(Y_test, Y_pred_ng)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred_ng))
mae = mean_absolute_error(Y_test, Y_pred_ng)
metrics['NGBoost'] = (r2, rmse, mae)

axs[6].scatter(Y_test, Y_pred_ng, edgecolors='k', alpha=0.7)
axs[6].plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
axs[6].set_title(f"NGBoost\nR²={r2:.2f} | RMSE={rmse:.2f} | MAE={mae:.2f}", fontsize=12)
axs[6].set_xlabel("Actual Hg", fontsize=10)
axs[6].set_ylabel("Predicted Hg", fontsize=10)
axs[6].text(0.05, 0.92, '(G)', transform=axs[6].transAxes, fontsize=14, fontweight='bold', ha='left', va='top')
axs[6].grid(True)

# TabNet Evaluation
Y_pred_tabnet = tabnet_model.predict(X_test_tab).ravel()
Y_pred_tabnet = transformer.inverse_transform(Y_pred_tabnet.reshape(-1, 1)).ravel()

r2 = r2_score(Y_test_tab, Y_pred_tabnet)
rmse = np.sqrt(mean_squared_error(Y_test_tab, Y_pred_tabnet))
mae = mean_absolute_error(Y_test_tab, Y_pred_tabnet)
metrics['TabNet'] = (r2, rmse, mae)

axs[7].scatter(Y_test_tab, Y_pred_tabnet, edgecolors='k', alpha=0.7)
axs[7].plot([Y_test_tab.min(), Y_test_tab.max()], [Y_test_tab.min(), Y_test_tab.max()], 'r--')
axs[7].set_title(f"TabNet\nR²={r2:.2f} | RMSE={rmse:.2f} | MAE={mae:.2f}", fontsize=12)
axs[7].set_xlabel("Actual Hg", fontsize=10)
axs[7].set_ylabel("Predicted Hg", fontsize=10)
axs[7].text(0.05, 0.92, '(H)', transform=axs[7].transAxes, fontsize=14, fontweight='bold', ha='left', va='top')
axs[7].grid(True)


# Stacking Model Evaluation
stacking = StackingRegressor(
    estimators=[(k, v) for k, v in models.items()],
    final_estimator=Ridge()
)

stacking.fit(X_train, Y_train_trans)
Y_pred_trans = stacking.predict(X_test)
Y_pred = transformer.inverse_transform(Y_pred_trans.reshape(-1, 1)).ravel()
r2 = r2_score(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
mae = mean_absolute_error(Y_test, Y_pred)
metrics['Stacking'] = (r2, rmse, mae)

axs[8].scatter(Y_test, Y_pred, edgecolors='k', alpha=0.7)
axs[8].plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
axs[8].set_title(f"Stacking\nR²={r2:.2f} | RMSE={rmse:.2f} | MAE={mae:.2f}", fontsize=12)
axs[8].set_xlabel("Actual Hg", fontsize=10)
axs[8].set_ylabel("Predicted Hg", fontsize=10)
axs[8].text(0.05, 0.92, '(I)', transform=axs[8].transAxes, fontsize=14, fontweight='bold', ha='left', va='top')
axs[8].grid(True)

plt.tight_layout()
save_path = r"D:\python Fig"
os.makedirs(save_path, exist_ok=True)
filename = os.path.join(save_path, "model_comparison_upgraded.svg")
plt.savefig(filename, format='svg')
plt.show()

# Print Evaluation Results
print("\n--- Model Evaluation Results（Hg in original units） ---")
for name, (r2, rmse, mae) in metrics.items():
    print(f"{name:12s} | R²: {r2:.3f} | RMSE: {rmse:.3f} | MAE: {mae:.3f}")
print(f"\n✅ Figure saved as SVG: {filename}")
