# Bank Customer Churn Prediction
An end-to-end machine learning system built with TypeScript and TensorFlow.js to predict customer churn, achieving 56% accuracy with 60% recall after discovering and resolving critical data leakage.

## Project Overview
Predicts bank customer churn using deep learning, enabling proactive retention strategies. The project demonstrates the complete ML lifecycle from data collection through model deployment.

### Key Achievement
Discovered and resolved **critical data leakage**  reducing inflated 99% accuracy to a realistic 56% predictive model using only pre-churn demographic data.

## Results
- **Accuracy:** 56% (realistic performance without leakage)
- **Recall:** 60% (catches 6 out of 10 churning customers)
- **Precision:** 26%
- **F1-Score:** 36%

*Note: Performance metrics reflect honest predictive capability using only data available before customer churn.*

## Tech Stack
**Machine Learning:**
- TensorFlow.js (tfjs-node)
- Deep Neural Network (256→128→64→1)
- Adam Optimizer, Binary Cross-Entropy Loss

**Data Processing:**
- Danfo.js (DataFrame operations)
- StandardScaler normalization
- One-hot encoding for categorical variables

**Development:**
- TypeScript
- Node.js
- Express (API - coming soon)

## Architecture
```
bank-customer-churn/
├── src/
│   ├── data/
│   │   ├── loader.ts          # Data ingestion
│   │   ├── preprocessor.ts    # Feature engineering
│   │   └── splitter.ts        # Train/val/test split
│   ├── models/
│   │   ├── churnModel.ts      # Neural network architecture
│   │   └── trainer.ts         # Training pipeline
│   └── config/
├── data/                       # Raw & processed data
└── models/                     # Saved model artifacts
```

## Getting Started
### Prerequisites
- Node.js 18+
- npm or yarn

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/bank-customer-churn.git
cd bank-customer-churn

# Install dependencies
npm install

# Setup environment
cp .env.example .env
```

### Download Dataset

1. Download from [Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)
2. Place `Customer-Churn-Records.csv` in `data/raw/`

### Run Training Pipeline
```bash
# Preprocess data
npm run preprocess

# Split into train/val/test
npm run split

# Train model
npm run train
```

## Model Architecture
```typescript
Input (17 features)
    ↓
Dense (256 neurons, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (0.5)
    ↓
Dense (128 neurons, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (0.4)
    ↓
Dense (64 neurons, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (0.3)
    ↓
Output (1 neuron, Sigmoid)
```

## Key Learnings

### Data Leakage Discovery
Initial model achieved 99% accuracy, but investigation revealed the "Complain" feature had 99.5% correlation with churn:
- 99.51% of complainers churned
- 99.80% of churners complained

This feature represents information available *after* the decision to churn, making it unsuitable for prediction.

### Solution
Removed leaky features (Complain, Satisfaction Score, Point Earned) and rebuilt model using only:
- Demographics (Age, Gender, Geography)
- Account information (Balance, CreditScore, Tenure)
- Product usage (NumOfProducts, HasCrCard, IsActiveMember)

Result: More realistic 56% accuracy with actionable predictions.

## Features Used

**Continuous Features (Normalized):**
- CreditScore, Age, Tenure, Balance
- EstimatedSalary, NumOfProducts
- Tenure_Age_Ratio, Balance_Per_Product

**Categorical Features (Encoded):**
- Geography (France, Germany, Spain)
- Gender (Male, Female)
- Card Type (Diamond, Gold, Platinum, Silver)

**Binary Features:**
- HasCrCard, IsActiveMember

## ML Techniques Applied

- **StandardScaler (Z-score normalization)** for feature scaling
- **Class weight balancing** (3.9:1) for imbalanced dataset
- **Dropout regularization** to prevent overfitting
- **Batch normalization** for training stability
- **Early stopping** to prevent overtraining
- **Threshold optimization** via ROC analysis
- **Confusion matrix analysis** for performance evaluation

## 📝 Future Improvements

- [ ] Deploy REST API with Express
- [ ] Build React dashboard for predictions
- [ ] Add model monitoring and retraining pipeline
- [ ] Implement SHAP values for explainability
- [ ] Try ensemble methods (XGBoost, Random Forest)
- [ ] Deploy to AWS Lambda or SageMaker

## 📄 License

MIT License - see LICENSE file

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.
This work is an open progress. Full upload coming soon

## 👤 Author
Axel

---

⭐ Star this repo if you found it helpful!
