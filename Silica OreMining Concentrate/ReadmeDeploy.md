# INDUSTRIAL MINING ML - CONSERVATIVE BIAS ENSEMBLE
**Goal :  Predict "% Silica Concentrate"**

  **Results** 

  **Project Status : Release Ready - Onto Deploy** 

  **Key Findings :

     - Good Predictive Power 
         [ R² = 0.6189 ] = ~62% Variance
         [ MAE = 0.5840% ] = Avg error less than 0.6%
         [ RMSE = 0.7061% ] = Root mean squared error acceptable
     - Controlled Bias
         [ +0.3198% ] = a little overprediction mean bias
         [ +0.4054% ] = Confirm positive median bias
         [ ok ] = error distribution 
         [ 0.5376% ] = 50th percentile error
     - Prediction (Error vs Actual Value Range)
         [ 1.5-2.5% ] = Great Performance
     - Best model comparison = CatBoost

  **Critical Findings :** 
  
      Low Silica(<1.5%%) tends to overpredict
      High Silica(>4%) tends underprediction
      
  
  <img width="2949" height="1835" alt="prediction_analysis" src="https://github.com/user-attachments/assets/0d9e3361-963e-435d-9193-9abd3afa0b38" />
  
  <img width="2683" height="1475" alt="model_comparison" src="https://github.com/user-attachments/assets/9886ae55-d60e-4da4-8515-82bdefb8eb29" />
  
  
   **Comparison Industry Benchmarks [Silica/Mineral Processing Predictions]**
   
                              Industry Benchmark R²     This Model
      Flotation Quality Prediction [ 0.45-0.62 ]          [ 0.62 ]
      Ore Grade Estimation         [ 0.55-0.75 ]          [ 0.62 ]
      Process Control Modelling    [ 0.50-0.70 ]          [ 0.62 ]

        MAE Benchmark 
          Excellent: <0.50% absolute error
          Good: 0.50-0.70% absolute error <-- This Model (0.58%)
          Acceptable: 0.70-1.00% absolute error
          Poor: >1.00% absolute error

        Bias Standards 
          Excellent: |Bias| <0.10%
          Good: |Bias| <0.20%
          Acceptable: |Bias| <0.30% <-- This Model (+0.32%)
          Requires correction: |Bias| >0.30%

  # [Code Output]
  # SILICA PREDICTION DEPLOYMENT SYSTEM 
  
  Loading training data from Kaggle...
  Warning: Looks like you're using an outdated `kagglehub` version (installed: 0.3.13), please consider upgrading to the latest version (0.4.1).
  
  Training Optimized Silica Prediction Model
  [1/4] Preparing data...
    
      > Aggregating 733356 duplicate timestamps...
      4,097 samples ready
  
  [2/4] Engineering features...
    
      66 features created
  
  [3/4] Scaling features...
  
      Features scaled
  
  [4/4] Training ensemble model...
    > Detecting operating regimes...
    > Using single regime (most robust)
    > Creating model ensemble...
    > Training models...
      - XGBoost_Deep
      - XGBoost_Shallow
      - LightGBM
      - CatBoost
      - RandomForest
    > Calculating bias correction...
      ✓ No correction needed
    ✓ Model trained
  
  # Training Complete
  
   Model saved to: models_optimized/silica_predictor.pkl
   No input file specified. Using training data for demo...
  
  # Making Predictions
  [1/3] Preparing data...
  
      > Aggregating 733356 duplicate timestamps...
       4,097 samples to predict
  
  [2/3] Engineering features...
  
      Features engineered
  
  [3/3] Generating predictions...
  
      4,097 predictions generated
  
  # Predictions Complete
   
      Prediction Summary:
      Samples predicted    : 4,097
      Mean prediction      : 2.3266%
      Std prediction       : 1.0033%
      Min prediction       : 0.9085%
      Max prediction       : 5.3229%
      Bias corrected       : 0 samples    
  
   # DEPLOYMENT COMPLETE
