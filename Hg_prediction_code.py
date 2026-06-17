import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import RidgeCV
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from ngboost import NGBRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from lightgbm import early_stopping, log_evaluation
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

os.environ["LIGHTGBM_VERBOSE"] = "0"
warnings.filterwarnings('ignore')

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_FILE = 'Supplementary Data S2.xlsx'
DATA_SHEETS = [
    'Jizhentun section', 'Xiahuayuan section', 'This study', 'Wuhe section', 'drill-core X1',
    'Wayao  section', 'Laishike section', 'Drewer WA section', 'Duli section', 'Dupont GHS section', 'Nandong section'
]
SAVE_DIR = 'output_figures'
RANDOM_STATE = 42

# ─── Optimised hyperparameters ────────────────────────────────────────────────
BEST_PARAMS = {
    'RandomForest': {
        'n_estimators': 94, 'max_depth': 7,
        'min_samples_split': 6, 'min_samples_leaf': 4,
        'max_features': 0.7909,
    },
    'XGB': {
        'n_estimators': 67, 'max_depth': 5, 'learning_rate': 0.0545,
        'subsample': 0.7161, 'colsample_bytree': 0.7762, 'gamma': 4,
        'reg_alpha': 25, 'reg_lambda': 7, 'min_child_weight': 11,
    },
    'LGB': {
        'n_estimators': 96, 'num_leaves': 11, 'learning_rate': 0.0389,
        'reg_alpha': 12, 'reg_lambda': 7, 'min_child_samples': 24,
        'min_split_gain': 0.0739, 'subsample': 0.7535,
        'feature_fraction': 0.7352, 'max_depth': 4, 'min_data_in_leaf': 16,
    },
    'Cat': {
        'iterations': 123, 'depth': 4, 'learning_rate': 0.1060,
        'l2_leaf_reg': 18, 'subsample': 0.7720, 'random_strength': 2.2577,
        'min_data_in_leaf': 20,
    },
    'NGB': {
        'n_estimators': 51, 'learning_rate': 0.0336,
        'minibatch_frac': 0.3399, 'col_sample': 0.2963, 'tol': 0.0043,
    },
    'MLP': {
        'hidden_layer_sizes': (128, 64), 'alpha': 0.00449,
        'learning_rate_init': 0.01654, 'batch_size': 28,
    },
    'SVR': {
        'C': 78, 'gamma': 0.002999, 'epsilon': 0.02006,
    },
    'TabNet': {
        'n_d': 150, 'n_a': 256, 'n_steps': 3,
        'gamma': 2, 'lambda_sparse': 1e-3,
        'optimizer_params': {'lr': 0.1},
        'batch_size': 47,
    },
}

# Global store for training-phase metrics (avoids re-loading models for plots)
model_train_results = {
    'train_r2':   {}, 'test_r2':   {},
    'train_pred': {}, 'test_pred': {},
    'train_true': {}, 'test_true': {},
    'train_rmse': {}, 'test_rmse': {},
}


# ─── 1. Data loading and preprocessing ───────────────────────────────────────
def load_data(test_size=0.2, random_state=RANDOM_STATE,
              use_robust_scaling=True, use_feature_selection=True):
    df = pd.concat(
        pd.read_excel(DATA_FILE, sheet_name=DATA_SHEETS).values(),
        ignore_index=True
    )
    df = df[df['Hg'].notna()]

    # Feature engineering
    df['TOC_TS']  = df['TOC'] * df['TS']
    df['Mo_Al']   = df['Mo']  / (df['Al']  + 1e-6)
    df['P_Al']    = df['P']   / (df['Al']  + 1e-6)
    df['Mo_TOC']  = df['Mo']  / (df['TOC'] + 1e-6)
    df['TS_Al']   = df['TS']  / (df['Al']  + 1e-6)
    df['TOC_Al']  = df['TOC'] / (df['Al']  + 1e-6)
    df['log_TOC'] = np.log1p(df['TOC'])
    df['log_Mo']  = np.log1p(df['Mo'])

    features = [
        'TOC', 'TS', 'Mo', 'P', 'Al',
        'TOC_TS', 'Mo_Al', 'P_Al', 'Mo_TOC', 'TS_Al', 'TOC_Al',
        'log_TOC', 'log_Mo',
    ]

    def iqr_mask(series, factor=2.0):
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        return (series >= Q1 - factor * IQR) & (series <= Q3 + factor * IQR)

    mask = iqr_mask(df['Hg'])
    for feat in features:
        if feat in df.columns:
            mask &= iqr_mask(df[feat])
    df = df[mask].reset_index(drop=True)


    scaler = RobustScaler() if use_robust_scaling else StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    if use_feature_selection:
        selector = SelectKBest(score_func=f_regression, k=min(12, len(features)))
        X_scaled = selector.fit_transform(X_scaled, df['Hg'].values)
        selected = [features[i] for i in selector.get_support(indices=True)]
        print(f"Selected features: {selected}")

    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    X_poly = poly.fit_transform(X_scaled)
    X_tab  = X_scaled.astype(np.float32)
    y      = df['Hg'].values

    y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
    X_poly_tr, X_poly_te, X_tab_tr, X_tab_te, y_tr, y_te = train_test_split(
        X_poly, X_tab, y,
        test_size=test_size, random_state=random_state, stratify=y_binned,
    )

    return {
        'X_poly_train': X_poly_tr, 'X_poly_test': X_poly_te,
        'X_tab_train':  X_tab_tr,  'X_tab_test':  X_tab_te,
        'y_train': y_tr,           'y_test':  y_te,
        'scaler': scaler,
    }


# ─── 2. Model factory ─────────────────────────────────────────────────────────
def get_model(name, data):
    base = {
        'RandomForest': (RandomForestRegressor,
                         {'random_state': RANDOM_STATE, 'n_jobs': -1}),
        'SVR':          (SVR,
                         {'kernel': 'rbf'}),
        'MLP':          (MLPRegressor,
                         {'max_iter': 2000, 'random_state': RANDOM_STATE,
                          'early_stopping': True, 'validation_fraction': 0.2}),
        'XGB':          (xgb.XGBRegressor,
                         {'objective': 'reg:squarederror', 'random_state': RANDOM_STATE,
                          'n_jobs': -1, 'tree_method': 'hist'}),
        'LGB':          (lgb.LGBMRegressor,
                         {'random_state': RANDOM_STATE, 'n_jobs': -1,
                          'force_row_wise': True, 'verbose': -1,
                          'boost_from_average': True, 'max_bin': 200}),
        'Cat':          (cb.CatBoostRegressor,
                         {'verbose': 0, 'random_state': RANDOM_STATE, 'thread_count': -1}),
        'NGB':          (NGBRegressor,
                         {'verbose': False, 'random_state': RANDOM_STATE,
                          'validation_fraction': 0.25}),
        'TabNet':       (TabNetRegressor,
                         {'seed': RANDOM_STATE}),
    }
    if name not in base:
        raise ValueError(f"Unknown model: {name}")

    cls, kwargs = base[name]
    params = BEST_PARAMS.get(name, {}).copy()

    if name == 'TabNet':
        params.pop('batch_size', None)

    model = cls(**{**kwargs, **params})
    X = data['X_tab_train'] if name == 'TabNet' else data['X_poly_train']
    return model, X, data['y_train']


# ─── 3. Training ──────────────────────────────────────────────────────────────
def train_model(name):
    print(f"\nTraining: {name}")
    data = load_data()
    model, X_full, y_full = get_model(name, data)

    val_size = 0.35 if name in ['NGB', 'Cat'] else 0.2
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=val_size, random_state=RANDOM_STATE
    )

    fit_kwargs = {}
    if name == 'XGB':
        fit_kwargs = {'eval_set': [(X_val, y_val)],
                      'early_stopping_rounds': 20, 'verbose': False}
    elif name == 'LGB':
        fit_kwargs = {'eval_set': [(X_val, y_val)],
                      'callbacks': [early_stopping(20), log_evaluation(0)]}
    elif name == 'Cat':
        fit_kwargs = {'eval_set': [(X_val, y_val)],
                      'early_stopping_rounds': 25,
                      'use_best_model': True, 'verbose': False}
    elif name == 'NGB':
        y_tr  = y_tr.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        fit_kwargs = {'X_val': X_val, 'Y_val': y_val, 'early_stopping_rounds': 25}
    elif name == 'TabNet':
        y_tr  = y_tr.reshape(-1, 1).astype(np.float32)
        y_val = y_val.reshape(-1, 1).astype(np.float32)
        fit_kwargs = {
            'eval_set': [(X_val, y_val)],
            'patience': 20, 'max_epochs': 120,
            'batch_size': BEST_PARAMS['TabNet']['batch_size'],
        }

    try:
        model.fit(X_tr, y_tr, **fit_kwargs)
    except Exception as e:
        print(f"Error training {name}: {e}")
        return None, 0, None

    joblib.dump(model, f"{name}_model.joblib")

    def predict_1d(m, X):
        p = m.predict(X)
        return p.ravel() if p.ndim > 1 else p

    y_tr_pred  = predict_1d(model, X_tr)
    r2_tr      = r2_score(y_tr, y_tr_pred)
    rmse_tr    = np.sqrt(mean_squared_error(y_tr, y_tr_pred))

    X_te = data['X_tab_test'] if name == 'TabNet' else data['X_poly_test']
    y_te = data['y_test']
    y_te_pred  = predict_1d(model, X_te)
    r2_te      = r2_score(y_te, y_te_pred)
    rmse_te    = np.sqrt(mean_squared_error(y_te, y_te_pred))
    delta      = r2_tr - r2_te

    print(f"  Train R² = {r2_tr:.4f}  |  Test R² = {r2_te:.4f}  |  "
          f"Train RMSE = {rmse_tr:.4f}  |  Test RMSE = {rmse_te:.4f}  |  "
          f"ΔR² = {delta:.4f}")
    print(f"  Saved: {name}_model.joblib")

    model_train_results['train_r2'][name]   = r2_tr
    model_train_results['test_r2'][name]    = r2_te
    model_train_results['train_pred'][name] = y_tr_pred
    model_train_results['test_pred'][name]  = y_te_pred
    model_train_results['train_true'][name] = y_tr
    model_train_results['test_true'][name]  = y_te
    model_train_results['train_rmse'][name] = rmse_tr
    model_train_results['test_rmse'][name]  = rmse_te

    return model, r2_te, y_te_pred


# ─── 4. Stacking ensemble ─────────────────────────────────────────────────────
def create_stacking_ensemble(base_names=('RandomForest', 'Cat', 'NGB')):
    print("\nBuilding stacking ensemble...")
    data    = load_data()
    X_tr    = data['X_poly_train']
    X_te    = data['X_poly_test']
    y_tr    = data['y_train']
    y_te    = data['y_test']

    kf          = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    meta_train  = np.zeros((X_tr.shape[0], len(base_names)))
    meta_test   = np.zeros((X_te.shape[0], len(base_names)))

    for i, bname in enumerate(base_names):
        print(f"  Base model: {bname}")
        m, _, _ = get_model(bname, data)
        fold_te  = []
        for tr_idx, val_idx in kf.split(X_tr):
            Xf_tr, Xf_val = X_tr[tr_idx], X_tr[val_idx]
            yf_tr         = y_tr[tr_idx]
            if bname in ['NGB', 'TabNet']:
                yf_tr = yf_tr.reshape(-1, 1)
            m.fit(Xf_tr, yf_tr)
            p = m.predict(Xf_val)
            meta_train[val_idx, i] = p.ravel() if p.ndim > 1 else p
            fold_te.append(m.predict(X_te))
        meta_test[:, i] = np.mean(fold_te, axis=0)

    meta_model = RidgeCV()
    meta_model.fit(meta_train, y_tr)
    joblib.dump(meta_model, "Stacking_model.joblib")

    y_pred  = meta_model.predict(meta_test)
    r2_te   = r2_score(y_te, y_pred)
    print(f"  Stacking Test R² = {r2_te:.4f}")
    return y_pred, r2_te


# ─── 5. Visualisation ─────────────────────────────────────────────────────────
COLORS = {
    'RandomForest': '#2E86AB', 'XGB': '#A23B72', 'LGB': '#F18F01',
    'Cat': '#C73E1D',  'NGB': '#7209B7', 'MLP': '#06A77D',
    'SVR': '#005F73',  'TabNet': '#F77F00', 'Stacking': '#D62828',
}
LETTERS = list('abcdefghijkl')


def _apply_nature_style():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('default')
        plt.rcParams.update({
            'axes.facecolor': 'white',
            'grid.color': 'lightgray',
            'grid.linestyle': '--',
        })
    rcParams.update({
        'font.family': 'Arial', 'font.size': 9,
        'axes.linewidth': 0.8, 'axes.labelsize': 9,
        'xtick.labelsize': 8,  'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300, 'savefig.dpi': 300,
        'pdf.fonttype': 42,    'ps.fonttype': 42,
    })


def _scatter_ax(ax, y_true_tr, y_pred_tr, y_true_te, y_pred_te,
                color, label, letter):
    ax.scatter(y_true_te, y_pred_te, c=color, alpha=0.7, s=18,
               edgecolors='white', linewidths=0.4, label='Test')
    ax.scatter(y_true_tr, y_pred_tr, c=color, alpha=0.35, s=18,
               edgecolors='white', linewidths=0.3, marker='s', label='Train')

    lo = min(y_true_te.min(), y_pred_te.min(),
             y_true_tr.min(), y_pred_tr.min())
    hi = max(y_true_te.max(), y_pred_te.max(),
             y_true_tr.max(), y_pred_tr.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, alpha=0.8)

    margin = (hi - lo) * 0.05
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)

    r2_tr   = r2_score(y_true_tr, y_pred_tr)
    r2_te   = r2_score(y_true_te, y_pred_te)
    rmse_te = np.sqrt(mean_squared_error(y_true_te, y_pred_te))
    mae_te  = mean_absolute_error(y_true_te, y_pred_te)
    delta   = r2_tr - r2_te

    box_color  = '#ffebee' if delta > 0.2 else '#fff3e0' if delta > 0.1 else '#e8f5e9'
    edge_color = '#f44336' if delta > 0.2 else '#ff9800' if delta > 0.1 else '#4caf50'

    ax.text(0.04, 0.87,
            f'Train $R^2$ = {r2_tr:.4f}\nTest $R^2$ = {r2_te:.4f}\n'
            f'RMSE = {rmse_te:.4f}\nMAE = {mae_te:.2f}',
            transform=ax.transAxes, va='top', fontsize=6.5,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=box_color,
                      alpha=0.9, edgecolor=edge_color, lw=0.8))

    ax.set_title(label, fontweight='bold', pad=10)
    ax.set_xlabel('Observed Hg (ng/g)', fontsize=8)
    ax.set_ylabel('Predicted Hg (ng/g)', fontsize=8)
    ax.text(0.02, 0.98, letter, transform=ax.transAxes,
            fontweight='bold', fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='black', alpha=0.8, lw=0.5))
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(direction='in', which='both', top=True, right=True,
                   length=3, width=0.6)

    return r2_te, rmse_te, mae_te, r2_tr, delta


def visualize_predictions(model_names, ensemble_pred=None):
    _apply_nature_style()

    n = len(model_names) + (1 if ensemble_pred is not None else 0)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    figsize = (2.8 * cols if n <= 4 else 11, max(3.0, 3.0 * rows))

    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = np.array(axs).flatten()

    metrics = {}

    for i, name in enumerate(model_names):
        if name not in model_train_results['train_r2']:
            print(f"No results for {name}, skipping.")
            continue
        r2_te, rmse_te, mae_te, r2_tr, delta = _scatter_ax(
            axs[i],
            model_train_results['train_true'][name],
            model_train_results['train_pred'][name],
            model_train_results['test_true'][name],
            model_train_results['test_pred'][name],
            COLORS.get(name, '#333333'), name, LETTERS[i],
        )
        metrics[name] = (r2_te, rmse_te, mae_te, r2_tr, delta)

    if ensemble_pred is not None:
        idx  = len(model_names)
        data = load_data()
        y_te = data['y_test']
        y_tr = data['y_train']

        try:
            sm  = joblib.load("Stacking_model.joblib")
            mts = [joblib.load(f"{m}_model.joblib")
                   for m in ['RandomForest', 'Cat', 'NGB']
                   if m in model_names]
            X_tr_poly = data['X_poly_train']
            meta = np.column_stack([np.ravel(m.predict(X_tr_poly)) for m in mts])
            y_tr_pred = sm.predict(meta)
        except Exception:
            y_tr_pred = np.full_like(y_tr, y_tr.mean())

        r2_te, rmse_te, mae_te, r2_tr, delta = _scatter_ax(
            axs[idx], y_tr, y_tr_pred, y_te, ensemble_pred,
            COLORS['Stacking'], 'Stacking Ensemble', LETTERS[idx],
        )
        metrics['Stacking'] = (r2_te, rmse_te, mae_te, r2_tr, delta)

        legend_handles = [
            Line2D([0], [0], marker='s', color='w', label='Train',
                   markerfacecolor='black', markersize=6, alpha=0.35),
            Line2D([0], [0], marker='o', color='w', label='Test',
                   markerfacecolor='black', markersize=6, alpha=0.7),
        ]
        fig.legend(handles=legend_handles, loc='lower right', ncol=2,
                   frameon=False, fontsize=8, bbox_to_anchor=(0.98, 0.01))

    for j in range(n, len(axs)):
        axs[j].axis('off')

    plt.tight_layout(pad=2.0)

    os.makedirs(SAVE_DIR, exist_ok=True)
    base = os.path.join(SAVE_DIR, 'model_comparison')
    plt.savefig(base + '.svg', format='svg', bbox_inches='tight')
    plt.savefig(base + '.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(base + '.png', dpi=600, bbox_inches='tight')
    plt.show()
    print(f"Figures saved to {SAVE_DIR}/")

    # Performance summary
    print(f"\n{'Model':<15} {'Test R²':>8} {'Train R²':>9} "
          f"{'Test RMSE':>10} {'MAE':>7} {'ΔR²':>7}")
    print('-' * 60)
    for name, (r2_te, rmse_te, mae_te, r2_tr, delta) in metrics.items():
        flag = ('severe overfit' if delta > 0.2 else
                'mild overfit'   if delta > 0.1 else
                'good fit')
        print(f"{name:<15} {r2_te:>8.4f} {r2_tr:>9.4f} "
              f"{rmse_te:>10.4f} {mae_te:>7.3f} {delta:>7.4f}  {flag}")

    return metrics


# ─── 6. Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    MODEL_NAMES = ['RandomForest', 'XGB', 'LGB', 'Cat', 'NGB', 'MLP', 'SVR', 'TabNet']

    results = {}
    for m in MODEL_NAMES:
        try:
            _, r2, _ = train_model(m)
            results[m] = r2
        except Exception as e:
            print(f"Error training {m}: {e}")

    print("\nModel performance ranking:")
    for rank, (m, r2) in enumerate(
            sorted(results.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"  {rank}. {m}: R² = {r2:.4f}")

    ensemble_pred = None
    stacking_base = [m for m in ['RandomForest', 'Cat', 'NGB'] if m in results]
    if len(stacking_base) >= 2:
        ensemble_pred, _ = create_stacking_ensemble(stacking_base)
    else:
        print("Fewer than 2 base models available; skipping stacking.")

    trained = [m for m in MODEL_NAMES if m in model_train_results['test_r2']]
    visualize_predictions(trained, ensemble_pred=ensemble_pred)
