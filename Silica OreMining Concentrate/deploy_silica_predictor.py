"""
Silica Prediction Deployment Script
====================================

Deployment script for making predictions on new mining process data.
Uses the optimized ensemble model with conservative bias correction.

Usage:
    python deploy_silica_predictor.py --input <new_data.csv> --output <predictions.csv>

Author: Deployment Version
Date: 2026-01-30
"""

import os
import sys
import argparse
import warnings
from typing import Dict, Tuple, Optional
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import kagglehub

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================

class DeployConfig:
    """Deployment configuration matching training setup."""
    
    TARGET = '%_Silica_Concentrate'
    RANDOM_SEED = 42
    MIN_REGIME_SAMPLES = 1000
    CORRECTION_THRESHOLD = 0.5
    MAX_CORRECTION = 0.15
    APPLY_CORRECTION_QUANTILE = 0.75
    
    MODELS_DIR = 'models_optimized'
    RESULTS_DIR = 'results_optimized'

CONFIG = DeployConfig()


# ==========================================
# FEATURE ENGINEERING (IDENTICAL TO TRAINING)
# ==========================================

class OptimizedFeatureEngineer:
    """
    Feature engineering pipeline - must match training exactly.
    """
    
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Apply targeted feature engineering."""
        df = df.copy()
        
        # 1. Process physics features
        df = OptimizedFeatureEngineer._add_process_physics(df)
        
        # 2. Temporal features
        df = OptimizedFeatureEngineer._add_temporal_features(df)
        
        # 3. Operating conditions
        df = OptimizedFeatureEngineer._add_operating_conditions(df)
        
        return df.dropna()
    
    @staticmethod
    def _add_process_physics(df: pd.DataFrame) -> pd.DataFrame:
        """Metallurgical process relationships."""
        
        # Flotation efficiency
        airflow_cols = [c for c in df.columns if 'Air_Flow' in c]
        if airflow_cols and '%_Iron_Feed' in df.columns:
            total_airflow = df[airflow_cols].sum(axis=1)
            df['specific_airflow'] = total_airflow / (df['%_Iron_Feed'] + 1.0)
        
        # Silica selectivity indicators
        if '%_Silica_Feed' in df.columns and '%_Iron_Feed' in df.columns:
            df['iron_silica_ratio'] = df['%_Iron_Feed'] / (df['%_Silica_Feed'] + 1e-6)
            df['total_gangue'] = df['%_Iron_Feed'] + df['%_Silica_Feed']
        
        # pH impact
        if 'Ore_Pulp_pH' in df.columns:
            df['ph_optimal_deviation'] = np.abs(df['Ore_Pulp_pH'] - 10.0)
            df['ph_squared'] = df['Ore_Pulp_pH'] ** 2
            
            if '%_Silica_Feed' in df.columns:
                df['ph_silica_interaction'] = df['Ore_Pulp_pH'] * df['%_Silica_Feed']
        
        # Pulp density effects
        if 'Ore_Pulp_Density' in df.columns:
            df['density_squared'] = df['Ore_Pulp_Density'] ** 2
            
            if airflow_cols:
                df['density_flow_ratio'] = df['Ore_Pulp_Density'] / (total_airflow + 1.0)
        
        # Column-specific intensities
        for i in range(1, 8):
            col_air = f'Flotation_Column_{i:02d}_Air_Flow'
            col_level = f'Flotation_Column_{i:02d}_Level'
            
            if col_air in df.columns and col_level in df.columns:
                df[f'col{i}_intensity'] = df[col_air] * df[col_level]
        
        # Iron recovery indicator
        if '%_Iron_Concentrate' in df.columns and '%_Iron_Feed' in df.columns:
            df['iron_recovery'] = df['%_Iron_Concentrate'] / (df['%_Iron_Feed'] + 1e-6)
        
        return df
    
    @staticmethod
    def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Selective temporal features."""
        
        key_features = [
            '%_Iron_Feed', '%_Silica_Feed', 'Ore_Pulp_pH', 
            'Flotation_Column_01_Air_Flow', 'Flotation_Column_02_Air_Flow'
        ]
        
        for col in key_features:
            if col not in df.columns:
                continue
            
            # Short-term trend
            df[f'{col}_ma6'] = df[col].rolling(6, min_periods=1).mean()
            
            # Medium-term stability
            df[f'{col}_ma24'] = df[col].rolling(24, min_periods=1).mean()
            
            # Process volatility
            df[f'{col}_std6'] = df[col].rolling(6, min_periods=1).std().fillna(0)
            
            # Recent change
            df[f'{col}_delta'] = df[col].diff().fillna(0)
            
            # Critical lags
            df[f'{col}_lag1'] = df[col].shift(1).fillna(method='bfill')
        
        return df
    
    @staticmethod
    def _add_operating_conditions(df: pd.DataFrame) -> pd.DataFrame:
        """Binary/categorical operating regime indicators."""
        
        # High throughput indicator
        airflow_cols = [c for c in df.columns if 'Air_Flow' in c and 'ma' not in c]
        if airflow_cols:
            total_airflow = df[airflow_cols].sum(axis=1)
            airflow_median = total_airflow.median()
            df['high_throughput'] = (total_airflow > airflow_median).astype(int)
        
        # Process stability indicator
        if '%_Iron_Feed' in df.columns:
            iron_std = df['%_Iron_Feed'].rolling(12).std()
            df['stable_feed'] = (iron_std < iron_std.median()).astype(int)
        
        # High silica feed indicator
        if '%_Silica_Feed' in df.columns:
            silica_median = df['%_Silica_Feed'].median()
            df['high_silica_feed'] = (df['%_Silica_Feed'] > silica_median).astype(int)
        
        return df


# ==========================================
# REGIME DETECTION
# ==========================================

class RobustRegimeDetector:
    """Conservative regime detection."""
    
    def __init__(self, min_samples: int = 1000):
        self.min_samples = min_samples
        self.kmeans = None
        self.scaler = RobustScaler()
        self.use_regimes = False
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict regime labels."""
        if not self.use_regimes:
            return np.zeros(len(df), dtype=int)
        
        regime_features = self._extract_regime_features(df)
        regime_features_scaled = self.scaler.transform(regime_features)
        return self.kmeans.predict(regime_features_scaled)
    
    def _extract_regime_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract regime indicators."""
        features = []
        
        # Throughput level
        airflow_cols = [c for c in df.columns if 'Air_Flow' in c and 'ma' not in c]
        if airflow_cols:
            total_airflow = df[airflow_cols].sum(axis=1)
            features.append(total_airflow)
        
        # Feed stability
        if '%_Iron_Feed' in df.columns:
            iron_std = df['%_Iron_Feed'].rolling(24, min_periods=1).std().fillna(0)
            features.append(iron_std)
        
        # Average ore grade
        if '%_Iron_Feed' in df.columns:
            features.append(df['%_Iron_Feed'])
        
        if '%_Silica_Feed' in df.columns:
            features.append(df['%_Silica_Feed'])
        
        return np.column_stack(features) if features else np.zeros((len(df), 1))


# ==========================================
# OPTIMIZED ENSEMBLE
# ==========================================

class OptimizedEnsemble:
    """Deployment version of optimized ensemble."""
    
    def __init__(self):
        self.models = {}
        self.regime_detector = None
        self.correction_stats = {}
    
    def _predict_raw(self, X: pd.DataFrame, regimes: np.ndarray) -> np.ndarray:
        """Generate raw predictions from ensemble."""
        predictions = np.zeros(len(X))
        
        for regime_id in np.unique(regimes):
            regime_mask = (regimes == regime_id)
            X_regime = X[regime_mask]
            
            # Average predictions from all models
            regime_preds = []
            for name, model in self.models[regime_id].items():
                regime_preds.append(model.predict(X_regime))
            
            predictions[regime_mask] = np.mean(regime_preds, axis=0)
        
        return predictions
    
    def predict(self, X: pd.DataFrame, apply_correction: bool = True) -> np.ndarray:
        """Generate predictions with optional conservative correction."""
        
        # Determine regimes
        regimes = self.regime_detector.predict(X)
        
        # Get baseline predictions
        predictions = self._predict_raw(X, regimes)
        
        # Apply conservative conditional correction
        if apply_correction:
            for regime_id in np.unique(regimes):
                if self.correction_stats.get(regime_id) is None:
                    continue
                
                regime_mask = (regimes == regime_id)
                correction_info = self.correction_stats[regime_id]
                
                # Only correct high predictions
                high_pred_mask = predictions > correction_info['threshold']
                correction_mask = regime_mask & high_pred_mask
                
                if correction_mask.sum() > 0:
                    predictions[correction_mask] -= correction_info['correction']
        
        return predictions


# ==========================================
# DEPLOYMENT PIPELINE
# ==========================================

class SilicaPredictor:
    """Complete deployment pipeline for silica prediction."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model. If None, trains on latest data.
        """
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_path = model_path
        
    def train_model(self, training_data: pd.DataFrame) -> None:
        """
        Train model on provided data.
        
        Args:
            training_data: DataFrame with features and target
        """
        print("\n Training Optimized Silica Prediction Model")
        print("=" * 70)
        
        # Prepare data
        print("\n[1/4] Preparing data...")
        df = training_data.copy()
        
        # Standardize column names
        df.columns = [c.replace(' ', '_') for c in df.columns]
        target = CONFIG.TARGET.replace(' ', '_')
        
        # Handle duplicates if timestamp index
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.duplicated().any():
                print(f"  > Aggregating {df.index.duplicated().sum()} duplicate timestamps...")
                df = df.groupby(df.index).agg('median')
        
        df = df.dropna()
        print(f"  ✓ {len(df):,} samples ready")
        
        # Feature engineering
        print("\n[2/4] Engineering features...")
        df_engineered = OptimizedFeatureEngineer.engineer_features(df)
        print(f"  ✓ {len([c for c in df_engineered.columns if c != target])} features created")
        
        X = df_engineered.drop(columns=[target])
        y = df_engineered[target]
        
        # Store feature columns for deployment
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        print("\n[3/4] Scaling features...")
        self.scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        print("  ✓ Features scaled")
        
        # Train ensemble
        print("\n[4/4] Training ensemble model...")
        self.model = self._train_ensemble(X_scaled, y)
        print("  ✓ Model trained")
        
        print("\n" + "=" * 70)
        print(" Training Complete")
        print("=" * 70)
    
    def _train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> OptimizedEnsemble:
        """Train the optimized ensemble."""
        
        ensemble = OptimizedEnsemble()
        
        # Step 1: Regime detection
        print("  > Detecting operating regimes...")
        regime_detector = RobustRegimeDetector(min_samples=CONFIG.MIN_REGIME_SAMPLES)
        regime_detector.use_regimes = False  # Simplified for deployment
        
        # For deployment, use single regime (most robust)
        regimes = np.zeros(len(X), dtype=int)
        print("  > Using single regime (most robust)")
        
        # Step 2: Create diverse models
        print("  > Creating model ensemble...")
        models = {
            'XGBoost_Deep': XGBRegressor(
                n_estimators=300, learning_rate=0.03, max_depth=8,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                gamma=0.1, random_state=CONFIG.RANDOM_SEED,
                tree_method='hist', device='cuda'
            ),
            'XGBoost_Shallow': XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=4,
                min_child_weight=10, subsample=0.9, colsample_bytree=0.9,
                random_state=CONFIG.RANDOM_SEED, tree_method='hist', device='cuda'
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=400, learning_rate=0.04, max_depth=7,
                num_leaves=50, min_child_samples=30, subsample=0.8,
                colsample_bytree=0.8, random_state=CONFIG.RANDOM_SEED,
                device='cpu', verbosity=-1
            ),
            'CatBoost': CatBoostRegressor(
                iterations=300, learning_rate=0.05, depth=6,
                l2_leaf_reg=5, random_state=CONFIG.RANDOM_SEED,
                task_type='CPU', verbose=False
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, max_features='sqrt',
                random_state=CONFIG.RANDOM_SEED, n_jobs=-1
            )
        }
        
        # Step 3: Train models
        print("  > Training models...")
        for name, model in models.items():
            print(f"    - {name}")
            model.fit(X, y)
        
        ensemble.models[0] = models
        ensemble.regime_detector = regime_detector
        
        # Step 4: Calculate correction statistics
        print("  > Calculating bias correction...")
        predictions_train = np.zeros(len(X))
        for model in models.values():
            predictions_train += model.predict(X)
        predictions_train /= len(models)
        
        # Only correct high predictions
        high_pred_threshold = np.quantile(predictions_train, CONFIG.APPLY_CORRECTION_QUANTILE)
        high_pred_mask = predictions_train > high_pred_threshold
        
        if high_pred_mask.sum() > 0:
            high_pred_errors = predictions_train[high_pred_mask] - y.values[high_pred_mask]
            median_error = np.median(high_pred_errors)
            
            if median_error > CONFIG.CORRECTION_THRESHOLD:
                correction = min(median_error * 0.5, CONFIG.MAX_CORRECTION)
                ensemble.correction_stats[0] = {
                    'correction': correction,
                    'threshold': high_pred_threshold
                }
                print(f"    ✓ Correction: -{correction:.4f} for predictions > {high_pred_threshold:.2f}")
            else:
                ensemble.correction_stats[0] = None
                print(f"    ✓ No correction needed")
        else:
            ensemble.correction_stats[0] = None
        
        return ensemble
    
    def predict(self, new_data: pd.DataFrame, 
                apply_correction: bool = True) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            new_data: DataFrame with same features as training data
            apply_correction: Whether to apply bias correction
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print("\n Making Predictions")
        print("=" * 70)
        
        # Prepare data
        print("\n[1/3] Preparing data...")
        df = new_data.copy()
        df.columns = [c.replace(' ', '_') for c in df.columns]
        
        # Handle duplicates if timestamp index
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.duplicated().any():
                print(f"  > Aggregating {df.index.duplicated().sum()} duplicate timestamps...")
                df = df.groupby(df.index).agg('median')
        
        original_index = df.index
        print(f"  ✓ {len(df):,} samples to predict")
        
        # Feature engineering
        print("\n[2/3] Engineering features...")
        df_engineered = OptimizedFeatureEngineer.engineer_features(df)
        print(f"  ✓ Features engineered")
        
        # Ensure feature alignment
        missing_features = set(self.feature_columns) - set(df_engineered.columns)
        if missing_features:
            print(f"    Warning: Missing features: {missing_features}")
            for feat in missing_features:
                df_engineered[feat] = 0
        
        X = df_engineered[self.feature_columns]
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Generate predictions
        print("\n[3/3] Generating predictions...")
        predictions = self.model.predict(X_scaled, apply_correction=apply_correction)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'predicted_silica_concentrate': predictions
        }, index=X_scaled.index)
        
        # Add confidence indicators
        results['prediction_type'] = 'standard'
        if apply_correction and 0 in self.model.correction_stats:
            if self.model.correction_stats[0] is not None:
                corrected_mask = predictions < (predictions + self.model.correction_stats[0]['correction'])
                results.loc[corrected_mask, 'prediction_type'] = 'bias_corrected'
        
        print(f"  ✓ {len(results):,} predictions generated")
        
        print("\n" + "=" * 70)
        print(" Predictions Complete")
        print("=" * 70)
        
        return results
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to disk."""
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f" Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)
        
        self.model = model_package['model']
        self.scaler = model_package['scaler']
        self.feature_columns = model_package['feature_columns']
        
        print(f" Model loaded from: {filepath}")


# ==========================================
# MAIN DEPLOYMENT INTERFACE
# ==========================================

def main():
    """Main deployment script."""
    
    parser = argparse.ArgumentParser(
        description='Deploy silica prediction model on new data'
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'predict', 'train_and_predict'],
        default='train_and_predict',
        help='Operation mode'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input CSV file for prediction'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models_optimized/silica_predictor.pkl',
        help='Path to save/load model'
    )
    parser.add_argument(
        '--no-correction',
        action='store_true',
        help='Disable bias correction'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" SILICA PREDICTION DEPLOYMENT SYSTEM")
    print("="*70)
    
    predictor = SilicaPredictor()
    
    # Training mode
    if args.mode in ['train', 'train_and_predict']:
        print("\n Loading training data from Kaggle...")
        try:
            path = kagglehub.dataset_download(
                "edumagalhaes/quality-prediction-in-a-mining-process"
            )
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            training_data = pd.read_csv(
                os.path.join(path, csv_files[0]),
                decimal=',',
                parse_dates=['date'],
                index_col='date'
            )
            
            predictor.train_model(training_data)
            predictor.save_model(args.model)
            
        except Exception as e:
            print(f" Error loading training data: {e}")
            return
    
    # Prediction mode
    if args.mode in ['predict', 'train_and_predict']:
        if args.mode == 'predict':
            print(f"\n Loading model from: {args.model}")
            predictor.load_model(args.model)
        
        # Load new data
        if args.input:
            print(f"\n Loading new data from: {args.input}")
            try:
                new_data = pd.read_csv(
                    args.input,
                    decimal=',',
                    parse_dates=['date'] if 'date' in pd.read_csv(args.input, nrows=0).columns else None,
                    index_col='date' if 'date' in pd.read_csv(args.input, nrows=0).columns else None
                )
            except Exception as e:
                print(f" Error loading input data: {e}")
                return
        else:
            print("\n  No input file specified. Using training data for demo...")
            # Drop target column if it exists (may have different name after preprocessing)
            target_col = CONFIG.TARGET.replace(' ', '_')
            possible_target_cols = [target_col, '%_Silica_Concentrate', 'Silica_Concentrate_Percent']
            new_data = training_data.copy()
            for col in possible_target_cols:
                if col in new_data.columns:
                    new_data = new_data.drop(columns=[col])
                    break
        
        # Make predictions
        predictions = predictor.predict(
            new_data,
            apply_correction=not args.no_correction
        )
        
        # Save predictions
        print(f"\n Saving predictions to: {args.output}")
        predictions.to_csv(args.output)
        
        # Display summary
        print("\n Prediction Summary:")
        print(f"  Samples predicted    : {len(predictions):,}")
        print(f"  Mean prediction      : {predictions['predicted_silica_concentrate'].mean():.4f}%")
        print(f"  Std prediction       : {predictions['predicted_silica_concentrate'].std():.4f}%")
        print(f"  Min prediction       : {predictions['predicted_silica_concentrate'].min():.4f}%")
        print(f"  Max prediction       : {predictions['predicted_silica_concentrate'].max():.4f}%")
        print(f"  Bias corrected       : {(predictions['prediction_type'] == 'bias_corrected').sum():,} samples")
        
        print(f"\n Predictions saved to: {args.output}")
    
    print("\n" + "="*70)
    print(" DEPLOYMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()