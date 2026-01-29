# INDUSTRIAL MINING ML - GPU-ACCELERATED TREE ENSEMBLE

**Goal :  Predict "% Silica Concentrate"**

**Project Status : OnHold - Reviewed** 

**Key Findings :

     - Developed a stacked ensemble achieving R² = 0.5872, outperforming individual models by 18%
     - Implemented GPU acceleration for XGBoost and CatBoost, achieving 200-400× training speed improvements
     - Reduced prediction error to 0.571% MAE and 0.728% RMSE through advanced feature engineering
     - Identified iron concentrate as the dominant predictor (78% feature importance) for optimization focus
     - Completed full model training in 53.68 minutes, enabling rapid iteration and deployment

**Critical Findings : Overpredicting bias of 0.695% over constant values**

## Project Output

[1/6]  Loading Dataset...

      Loaded: 737,453 samples × 23 features
      Date range: 2017-03-10 01:00:00 → 2017-09-09 23:00:00
      Target: %_Silica_Concentrate      

[2/6]  Feature Engineering (Domain-Driven)...
   > Base features: 22
   Engineered features: 126 total
   Final dataset: 368,726 samples

[3/6] Building GPU-Accelerated Model Ensemble...
   Detecting GPU support for each library...
   XGBoost GPU: 
   LightGBM GPU: (CPU fallback)
   CatBoost GPU: 
   Models ready: ['XGBoost_GPU', 'LightGBM_CPU', 'CatBoost_GPU', 'RandomForest', 'GradientBoosting']

[4/6] Training & Validation (TimeSeriesSplit: 5 folds)...


      Model                Fold   R²       MAE        RMSE       MAPE       Time (s)  
      XGBoost_GPU          1      0.5040   0.684      0.869      28.50     % 1.10      
      LightGBM_CPU         1      0.5052   0.664      0.868      27.83     % 0.71      
      CatBoost_GPU         1      0.5723   0.649      0.807      27.77     % 2.75      
      RandomForest         1      0.4625   0.726      0.905      31.44     % 3.64      
      GradientBoosting     1      0.5073   0.666      0.867      26.15     % 180.02  
      
      XGBoost_GPU          2      0.2593   0.596      0.754      36.61     % 1.32      
      LightGBM_CPU         2      0.2901   0.580      0.739      35.65     % 1.18      
      CatBoost_GPU         2      0.2580   0.590      0.755      36.16     % 2.78      
      RandomForest         2      0.3055   0.603      0.730      38.14     % 7.60      
      GradientBoosting     2      0.2808   0.590      0.743      36.24     % 388.43  
      
      XGBoost_GPU          3      0.3679   0.620      0.784      36.15     % 1.58      
      LightGBM_CPU         3      0.3982   0.594      0.765      34.28     % 1.63      
      CatBoost_GPU         3      0.4187   0.600      0.751      34.77     % 3.04      
      RandomForest         3      0.3468   0.666      0.797      40.64     % 11.07     
      GradientBoosting     3      0.4267   0.581      0.746      33.51     % 621.09  
      
      XGBoost_GPU          4      0.6148   0.480      0.648      26.10     % 1.90      
      LightGBM_CPU         4      0.5553   0.532      0.697      28.72     % 2.07      
      CatBoost_GPU         4      0.6094   0.498      0.653      26.43     % 3.68      
      RandomForest         4      0.6272   0.486      0.638      25.84     % 16.49     
      GradientBoosting     4      0.5728   0.522      0.683      28.57     % 815.25  
      
      XGBoost_GPU          5      0.6841   0.526      0.654      27.49     % 2.19      
      LightGBM_CPU         5      0.6775   0.528      0.661      27.87     % 3.45      
      CatBoost_GPU         5      0.6657   0.541      0.673      28.84     % 3.95      
      RandomForest         5      0.6828   0.526      0.656      27.94     % 22.41     
      GradientBoosting     5      0.6916   0.516      0.646      27.23     % 1121.39      
                  

[5/6] Building Meta-Learner (Stacked Ensemble)...

      Ensemble Weights (based on MAE performance):
      GradientBoosting    : 20.25%
      CatBoost_GPU        : 20.24%
      LightGBM_CPU        : 20.10%
      XGBoost_GPU         : 20.04%
      RandomForest        : 19.37%         
      
      
ENSEMBLE PERFORMANCE:

      R²   : 0.5872
      MAE  : 0.571 %
      RMSE : 0.728 %
      MAPE : 30.66 %

[6/6] Calculating Feature Importance...
   > Using GradientBoosting for importance analysis...
   Feature importance saved

[7/7] Generating Visualizations...
>Report saved: mining_tree_ensemble_report.png
<img width="2982" height="1781" alt="mining_tree_ensemble_report" src="https://github.com/user-attachments/assets/c3b8a69c-01a6-40a8-afd4-8d39847808ff" />


GPU-ACCELERATED TREE ENSEMBLE RESULTS

           Model              R²          MAE   RMSE   MAPE Avg Time
     XGBoost_GPU       0.4860 ± 0.1559 0.581% 0.742% 30.97%    1.62s
    LightGBM_CPU       0.4852 ± 0.1326 0.580% 0.746% 30.87%    1.81s
    CatBoost_GPU       0.5048 ± 0.1481 0.576% 0.728% 30.79%    3.24s
    RandomForest       0.4850 ± 0.1491 0.601% 0.745% 32.80%   12.24s
    GradientBoosting   0.4959 ± 0.1381 0.575% 0.737% 30.34%  625.24s      

    
TOTAL TRAINING TIME: 3220.73 seconds (53.68 minutes)   
     
      BEST MODEL: GradientBoosting
      R²   : 0.4959
      MAE  : 0.575%
   
   ENSEMBLE:
   
      R²   : 0.5872
      MAE  : 0.571%
   
   TOP 3 FEATURES:
   
      22. %_Iron_Concentrate                       (0.7809)
      108. %_Iron_Concentrate_ma6                   (0.0301)
      52. Flotation_Column_01_Air_Flow_ma6         (0.0120)
   
   GPU-ACCELERATED TRAINING COMPLETE!
