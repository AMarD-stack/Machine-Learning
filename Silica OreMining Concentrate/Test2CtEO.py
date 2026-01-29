import os
import gc
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION - GPU ENABLED
# ==========================================
CONFIG = {
    'TARGET': '% Silica Concentrate',
    'N_SPLITS': 5,
    'RANDOM_SEED': 42,
    'USE_GPU': True  # Toggle GPU acceleration
}

np.random.seed(CONFIG['RANDOM_SEED'])

print("=" * 70)
print(" INDUSTRIAL MINING ML - GPU-ACCELERATED TREE ENSEMBLE")
print("=" * 70)
print(f" GPU Mode: {'ENABLED ' if CONFIG['USE_GPU'] else 'DISABLED (CPU)'}")

# ==========================================
# 1. DATA LOADING
# ==========================================
print("\n[1/6] Loading Dataset...")
try:
    path = kagglehub.dataset_download("edumagalhaes/quality-prediction-in-a-mining-process")
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV found")
    
    df = pd.read_csv(csv_files[0], decimal=',', parse_dates=['date'], index_col='date')
    df.columns = [c.replace(' ', '_') for c in df.columns]
    CONFIG['TARGET'] = CONFIG['TARGET'].replace(' ', '_')
    
    print(f" Loaded: {df.shape[0]:,} samples × {df.shape[1]} features")
    print(f"   Date range: {df.index.min()} → {df.index.max()}")
    print(f"   Target: {CONFIG['TARGET']}")

except Exception as e:
    print(f" Error: {e}")
    exit()

# ==========================================
# 2. INDUSTRIAL FEATURE ENGINEERING
# ==========================================
print("\n[2/6] Feature Engineering (Domain-Driven)...")

df = df.dropna()

# Store original feature names
base_features = [c for c in df.columns if c != CONFIG['TARGET']]
print(f"   > Base features: {len(base_features)}")

# --- Temporal Features (Process Inertia) ---
for col in base_features:
    df[f'{col}_ma3'] = df[col].rolling(window=3, min_periods=1).mean()
    df[f'{col}_ma6'] = df[col].rolling(window=6, min_periods=1).mean()
    df[f'{col}_std3'] = df[col].rolling(window=3, min_periods=1).std().fillna(0)
    df[f'{col}_delta'] = df[col].diff().fillna(0)

# --- Lag Features ---
lag_features = ['Flotation_Column_01_Air_Flow', 'Flotation_Column_02_Air_Flow', 
                'Flotation_Column_01_Level', 'Flotation_Column_02_Level',
                '%_Iron_Feed', '%_Silica_Feed']

for col in lag_features:
    if col in df.columns:
        df[f'{col}_lag1'] = df[col].shift(1).fillna(method='bfill')
        df[f'{col}_lag2'] = df[col].shift(2).fillna(method='bfill')
# Use only 50% of data for faster testing
df = df.sample(frac=0.5, random_state=42).sort_index()

# --- Physical Interaction Features ---
if 'Flotation_Column_01_Air_Flow' in df.columns and 'Flotation_Column_01_Level' in df.columns:
    df['col1_intensity'] = df['Flotation_Column_01_Air_Flow'] * df['Flotation_Column_01_Level']
    df['col2_intensity'] = df['Flotation_Column_02_Air_Flow'] * df['Flotation_Column_02_Level']

if '%_Iron_Feed' in df.columns and '%_Silica_Feed' in df.columns:
    df['iron_silica_ratio'] = df['%_Iron_Feed'] / (df['%_Silica_Feed'] + 1e-6)

airflow_cols = [c for c in df.columns if 'Air_Flow' in c and 'ma' not in c and 'std' not in c and 'delta' not in c]
if airflow_cols:
    df['total_airflow'] = df[airflow_cols].sum(axis=1)

df = df.dropna()

X = df.drop(columns=[CONFIG['TARGET']])
y = df[CONFIG['TARGET']]

print(f" Engineered features: {X.shape[1]} total")
print(f" Final dataset: {X.shape[0]:,} samples")

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# ==========================================
# 3. GPU-ACCELERATED MODEL FACTORY
# ==========================================
print("\n[3/6] 🏗️ Building GPU-Accelerated Model Ensemble...")

# Auto-detect GPU support for each library
def check_gpu_support():
    gpu_status = {'xgboost': False, 'lightgbm': False, 'catboost': False}
    
    # Check XGBoost
    try:
        import xgboost as xgb
        if 'gpu' in str(xgb.build_info()).lower() or 'cuda' in str(xgb.build_info()).lower():
            gpu_status['xgboost'] = True
    except:
        pass
    
    # Check LightGBM (check if CUDA version installed)
        # try:
        #     import lightgbm as lgb
        #     # Try creating a small GPU dataset to test
        #     test_data = lgb.Dataset(np.random.rand(10, 5), label=np.random.rand(10))
        #     try:
        #         lgb.train({'device': 'cuda'}, test_data, num_boost_round=1, verbose_eval=False)
        #         gpu_status['lightgbm'] = True
        #     except:
        #         try:
        #             lgb.train({'device': 'gpu'}, test_data, num_boost_round=1, verbose_eval=False)
        #             gpu_status['lightgbm'] = True
        #         except:
        #             pass
        # except:
        #     pass
    
    # CatBoost usually works with GPU if CUDA available
    try:
        from catboost import CatBoostRegressor
        test_model = CatBoostRegressor(iterations=1, task_type='GPU', verbose=False)
        test_model.fit(np.random.rand(10, 5), np.random.rand(10))
        gpu_status['catboost'] = True
    except:
        pass
    
    return gpu_status

if CONFIG['USE_GPU']:
    print(" Detecting GPU support for each library...")
    gpu_support = check_gpu_support()
    print(f"   XGBoost GPU: {'' if gpu_support['xgboost'] else ' (CPU fallback)'}")
    print(f"   LightGBM GPU: {'' if gpu_support['lightgbm'] else ' (CPU fallback)'}")
    print(f"   CatBoost GPU: {'' if gpu_support['catboost'] else ' (CPU fallback)'}")
else:
    gpu_support = {'xgboost': False, 'lightgbm': False, 'catboost': False}

if CONFIG['USE_GPU'] and any(gpu_support.values()):
    models = {
        'XGBoost_GPU' if gpu_support['xgboost'] else 'XGBoost_CPU': XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            random_state=CONFIG['RANDOM_SEED'],
            tree_method='hist' if gpu_support['xgboost'] else 'auto',
            device='cuda' if gpu_support['xgboost'] else 'cpu',
            verbosity=0
        ),
        
        'LightGBM_GPU' if gpu_support['lightgbm'] else 'LightGBM_CPU': LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=CONFIG['RANDOM_SEED'],
            device='cuda' if gpu_support['lightgbm'] else 'cpu',
            n_jobs=-1 if not gpu_support['lightgbm'] else 1,
            verbosity=-1
        ),
        
        'CatBoost_GPU' if gpu_support['catboost'] else 'CatBoost_CPU': CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            random_state=CONFIG['RANDOM_SEED'],
            task_type='GPU' if gpu_support['catboost'] else 'CPU',
            devices='0' if gpu_support['catboost'] else None,
            thread_count=-1 if not gpu_support['catboost'] else None,
            verbose=False
        ),
        
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=CONFIG['RANDOM_SEED'],
            n_jobs=-1
        ),
        
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            subsample=0.8,
            random_state=CONFIG['RANDOM_SEED']
        )
    }
else:
    print(" CPU mode selected")
    models = {
        'XGBoost': XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            random_state=CONFIG['RANDOM_SEED'],
            n_jobs=-1,
            verbosity=0
        ),
        
        'LightGBM': LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=CONFIG['RANDOM_SEED'],
            n_jobs=-1,
            verbosity=-1
        ),
        
        'CatBoost': CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            random_state=CONFIG['RANDOM_SEED'],
            verbose=False,
            thread_count=-1
        ),
        
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=CONFIG['RANDOM_SEED'],
            n_jobs=-1
        ),
        
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            subsample=0.8,
            random_state=CONFIG['RANDOM_SEED']
        )
    }

print(f" Models ready: {list(models.keys())}")

# ==========================================
# 4. TIME-SERIES CROSS-VALIDATION
# ==========================================
print(f"\n[4/6] Training & Validation (TimeSeriesSplit: {CONFIG['N_SPLITS']} folds)...")
print("\n" + "─" * 100)
print(f"{'Model':<20} {'Fold':<6} {'R²':<8} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'Time (s)':<10}")
print("─" * 100)

tscv = TimeSeriesSplit(n_splits=CONFIG['N_SPLITS'])

results = {name: {
    'r2': [], 'mae': [], 'rmse': [], 'mape': [],
    'predictions': [], 'actuals': [], 'train_time': []
} for name in models.keys()}

meta_X_list = []
meta_y_list = []

import time

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
    X_train = X_scaled.iloc[train_idx]
    X_val = X_scaled.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    
    fold_predictions = []
    
    for name, model in models.items():
        start_time = time.time()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        train_time = time.time() - start_time
        
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mape = mean_absolute_percentage_error(y_val, y_pred) * 100
        
        results[name]['r2'].append(r2)
        results[name]['mae'].append(mae)
        results[name]['rmse'].append(rmse)
        results[name]['mape'].append(mape)
        results[name]['train_time'].append(train_time)
        results[name]['predictions'].extend(y_pred)
        results[name]['actuals'].extend(y_val.values)
        
        fold_predictions.append(y_pred)
        
        print(f"{name:<20} {fold+1:<6} {r2:<8.4f} {mae:<10.3f} {rmse:<10.3f} {mape:<10.2f}% {train_time:<10.2f}")
    
    meta_X_list.append(np.column_stack(fold_predictions))
    meta_y_list.append(y_val.values)

print("─" * 100)

# ==========================================
# 5. META-LEARNER
# ==========================================
print("\n[5/6] Building Meta-Learner (Stacked Ensemble)...")

meta_X = np.vstack(meta_X_list)
meta_y = np.concatenate(meta_y_list)

weights = {}
for name in models.keys():
    avg_mae = np.mean(results[name]['mae'])
    weights[name] = 1.0 / avg_mae

total_weight = sum(weights.values())
weights = {k: v/total_weight for k, v in weights.items()}

print("\n Ensemble Weights (based on MAE performance):")
for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
    print(f"   {name:<20}: {weight:>6.2%}")

ensemble_pred = np.zeros_like(meta_y)
for i, name in enumerate(models.keys()):
    ensemble_pred += weights[name] * meta_X[:, i]

ensemble_r2 = r2_score(meta_y, ensemble_pred)
ensemble_mae = mean_absolute_error(meta_y, ensemble_pred)
ensemble_rmse = np.sqrt(mean_squared_error(meta_y, ensemble_pred))
ensemble_mape = mean_absolute_percentage_error(meta_y, ensemble_pred) * 100

print(f"\n ENSEMBLE PERFORMANCE:")
print(f"   R²   : {ensemble_r2:.4f}")
print(f"   MAE  : {ensemble_mae:.3f} %")
print(f"   RMSE : {ensemble_rmse:.3f} %")
print(f"   MAPE : {ensemble_mape:.2f} %")

# ==========================================
# 6. FEATURE IMPORTANCE
# ==========================================
print("\n[6/6] Calculating Feature Importance...")

best_model_name = min(results.keys(), key=lambda k: np.mean(results[k]['mae']))
print(f"   > Using {best_model_name} for importance analysis...")

best_model = models[best_model_name]
best_model.fit(X_scaled, y)

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
elif hasattr(best_model, 'get_feature_importance'):
    importances = best_model.get_feature_importance()
else:
    importances = np.zeros(len(X.columns))

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

feature_importance.to_json('feature_importance_tree_ensemble.json', orient='records', indent=2)

print("Feature importance saved")

# ==========================================
# 7. VISUALIZATION
# ==========================================
print("\n[7/7] Generating Visualizations...")

fig = plt.figure(figsize=(20, 12))
plt.style.use('seaborn-v0_8-darkgrid')

ax1 = plt.subplot(3, 3, 1)
model_names = list(results.keys())
r2_means = [np.mean(results[k]['r2']) for k in model_names]
r2_stds = [np.std(results[k]['r2']) for k in model_names]

bars = ax1.barh(model_names, r2_means, xerr=r2_stds, capsize=5, 
                color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
ax1.set_xlabel('R² Score', fontweight='bold')
ax1.set_title('Model Performance (R² Score)', fontweight='bold', fontsize=12)
ax1.set_xlim(0, 1)
for i, (bar, mean) in enumerate(zip(bars, r2_means)):
    ax1.text(mean + 0.02, bar.get_y() + bar.get_height()/2, 
             f'{mean:.4f}', va='center', fontsize=9)

ax2 = plt.subplot(3, 3, 2)
mae_means = [np.mean(results[k]['mae']) for k in model_names]
mae_stds = [np.std(results[k]['mae']) for k in model_names]

bars = ax2.barh(model_names, mae_means, xerr=mae_stds, capsize=5,
                color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
ax2.set_xlabel('MAE (% Silica)', fontweight='bold')
ax2.set_title('Mean Absolute Error', fontweight='bold', fontsize=12)
for i, (bar, mean) in enumerate(zip(bars, mae_means)):
    ax2.text(mean + 0.02, bar.get_y() + bar.get_height()/2, 
             f'{mean:.3f}%', va='center', fontsize=9)

ax3 = plt.subplot(3, 3, 3)
time_means = [np.mean(results[k]['train_time']) for k in model_names]
bars = ax3.bar(range(len(model_names)), time_means, 
               color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
ax3.set_xticks(range(len(model_names)))
ax3.set_xticklabels(model_names, rotation=45, ha='right')
ax3.set_ylabel('Time (seconds)', fontweight='bold')
ax3.set_title('Training Speed (GPU vs CPU)', fontweight='bold', fontsize=12)
for bar, time in zip(bars, time_means):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{time:.2f}s', ha='center', fontsize=9)

ax4 = plt.subplot(3, 3, 4)
best_preds = results[best_model_name]['predictions'][-500:]
best_actuals = results[best_model_name]['actuals'][-500:]

ax4.plot(best_actuals, label='Actual', color='black', linewidth=2, alpha=0.7)
ax4.plot(best_preds, label=f'{best_model_name} Prediction', 
         color='#e74c3c', linestyle='--', linewidth=1.5)
ax4.set_xlabel('Sample Index', fontweight='bold')
ax4.set_ylabel('% Silica Concentrate', fontweight='bold')
ax4.set_title(f'Process Control ({best_model_name})', fontweight='bold', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5 = plt.subplot(3, 3, 5)
ensemble_preds_plot = ensemble_pred[-300:]
actuals_plot = meta_y[-300:]
best_single_plot = results[best_model_name]['predictions'][-300:]

ax5.plot(actuals_plot, label='Actual', color='black', linewidth=2.5, alpha=0.8)
ax5.plot(best_single_plot, label=f'Best Single', 
         color='#3498db', linestyle='--', linewidth=1.5, alpha=0.7)
ax5.plot(ensemble_preds_plot, label='Ensemble', 
         color='#2ecc71', linestyle='-.', linewidth=2)
ax5.set_xlabel('Sample Index', fontweight='bold')
ax5.set_ylabel('% Silica Concentrate', fontweight='bold')
ax5.set_title('Ensemble vs Single Model', fontweight='bold', fontsize=12)
ax5.legend()
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(3, 3, 6)
residuals = np.array(best_actuals) - np.array(best_preds)
ax6.hist(residuals, bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
ax6.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax6.set_xlabel('Prediction Error (% Silica)', fontweight='bold')
ax6.set_ylabel('Frequency', fontweight='bold')
ax6.set_title(f'Error Distribution', fontweight='bold', fontsize=12)
ax6.text(0.02, 0.95, f'Mean: {np.mean(residuals):.3f}\nStd: {np.std(residuals):.3f}',
         transform=ax6.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax7 = plt.subplot(3, 3, 7)
top_features = feature_importance.head(20)
colors_feat = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
ax7.barh(range(len(top_features)), top_features['importance'], color=colors_feat)
ax7.set_yticks(range(len(top_features)))
ax7.set_yticklabels(top_features['feature'], fontsize=8)
ax7.set_xlabel('Importance Score', fontweight='bold')
ax7.set_title(f'Top 20 Features', fontweight='bold', fontsize=12)
ax7.invert_yaxis()

ax8 = plt.subplot(3, 3, 8)
ax8.scatter(best_actuals, best_preds, alpha=0.5, s=10, color='#3498db')
min_val = min(min(best_actuals), min(best_preds))
max_val = max(max(best_actuals), max(best_preds))
ax8.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
ax8.set_xlabel('Actual % Silica', fontweight='bold')
ax8.set_ylabel('Predicted % Silica', fontweight='bold')
ax8.set_title('Prediction Accuracy', fontweight='bold', fontsize=12)
ax8.legend()
ax8.grid(True, alpha=0.3)

ax9 = plt.subplot(3, 3, 9)
for name in model_names[:3]:
    fold_r2s = results[name]['r2']
    ax9.plot(range(1, len(fold_r2s)+1), fold_r2s, marker='o', label=name, linewidth=2)

ax9.set_xlabel('Fold Number', fontweight='bold')
ax9.set_ylabel('R² Score', fontweight='bold')
ax9.set_title('Cross-Validation Stability', fontweight='bold', fontsize=12)
ax9.legend()
ax9.grid(True, alpha=0.3)
ax9.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('mining_tree_ensemble_report.png', dpi=150, bbox_inches='tight')
print("Report saved: mining_tree_ensemble_report.png")

# ==========================================
# FINAL SUMMARY
# ==========================================
print("\n" + "=" * 70)
print("GPU-ACCELERATED TREE ENSEMBLE RESULTS")
print("=" * 70)

summary_data = []
for name in model_names:
    summary_data.append({
        'Model': name,
        'R²': f"{np.mean(results[name]['r2']):.4f} ± {np.std(results[name]['r2']):.4f}",
        'MAE': f"{np.mean(results[name]['mae']):.3f}%",
        'RMSE': f"{np.mean(results[name]['rmse']):.3f}%",
        'MAPE': f"{np.mean(results[name]['mape']):.2f}%",
        'Avg Time': f"{np.mean(results[name]['train_time']):.2f}s"
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

total_time = sum([sum(results[k]['train_time']) for k in model_names])
print("\n" + "─" * 70)
print(f" TOTAL TRAINING TIME: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f" BEST MODEL: {best_model_name}")
print(f"   R²   : {np.mean(results[best_model_name]['r2']):.4f}")
print(f"   MAE  : {np.mean(results[best_model_name]['mae']):.3f}%")

print(f"\n ENSEMBLE:")
print(f"   R²   : {ensemble_r2:.4f}")
print(f"   MAE  : {ensemble_mae:.3f}%")

print(f"\n TOP 3 FEATURES:")
for i, row in feature_importance.head(3).iterrows():
    print(f"   {i+1}. {row['feature']:<40} ({row['importance']:.4f})")

print("\n" + "=" * 70)
print(" GPU-ACCELERATED TRAINING COMPLETE!")
print("=" * 70)