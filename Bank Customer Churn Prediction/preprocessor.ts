import * as dfd from 'danfojs-node';
import * as path from 'path';
import * as fs from 'fs';
import { config } from 'dotenv';

config();

/**
 * DataPreprocessor - Handles data cleaning, transformation, and feature engineering
 * Following ML Lifecycle: Data Preparation phase
 */
export class DataPreprocessor {
  private df: dfd.DataFrame;

  constructor(df: dfd.DataFrame) {
    this.df = df;
  }

  /**
   * Check for missing values across all columns
   */
  checkMissingValues(): void {
    console.log('\n Checking for missing values');
    console.log('─'.repeat(50));
    
    const columns = this.df.columns;
    let hasMissing = false;

    columns.forEach((col: string) => {
      const nullCount = this.df.column(col).isNa().sum();
      if (nullCount > 0) {
        console.log(`${col}: ${nullCount} missing values`);
        hasMissing = true;
      }
    });

    if (!hasMissing) {
      console.log(' No missing values found!');
    }
  }

  /**
   * Remove unnecessary columns that don't contribute to prediction
   */
  dropUnnecessaryColumns(): void {
    console.log('\n  Dropping unnecessary columns...');
    
    // Columns to drop: RowNumber, CustomerId, Surname (identifiers, not features)
    const columnsToDrop = ['RowNumber', 'CustomerId', 'Surname'];
    
    columnsToDrop.forEach(col => {
      if (this.df.columns.includes(col)) {
        this.df = this.df.drop({ columns: [col] });
        console.log(`   Dropped: ${col}`);
      }
    });
    
    console.log(` Shape after dropping: ${this.df.shape[0]} rows × ${this.df.shape[1]} columns`);
  }

  /**
   * Encode categorical variables
   * - Binary encoding for Gender (Male/Female → 0/1)
   * - One-hot encoding for Geography and Card Type
   */
  encodeCategoricalVariables(): void {
    console.log('\n Encoding categorical variables');
    
    // 1. Binary encode Gender
    if (this.df.columns.includes('Gender')) {
      const genderMap = { 'Male': 1, 'Female': 0 };
      const genderEncoded = this.df.column('Gender').map((val: any) => genderMap[val as keyof typeof genderMap] ?? 0);
      this.df.addColumn('Gender_Encoded', genderEncoded);
      this.df = this.df.drop({ columns: ['Gender'] });
      console.log('   ✓ Gender encoded (Male=1, Female=0)');
    }

    // 2. One-hot encode Geography
    if (this.df.columns.includes('Geography')) {
      const geoDummies = dfd.getDummies(this.df.column('Geography'), { prefix: 'Geography' });
      
      // Add each dummy column
      geoDummies.columns.forEach((col: string) => {
        this.df.addColumn(col, geoDummies.column(col));
      });
      
      this.df = this.df.drop({ columns: ['Geography'] });
      console.log(` Geography one-hot encoded (${geoDummies.columns.length} columns)`);
    }

    // 3. One-hot encode Card Type
    if (this.df.columns.includes('Card Type')) {
      const cardDummies = dfd.getDummies(this.df.column('Card Type'), { prefix: 'CardType' });
      
      cardDummies.columns.forEach((col: string) => {
        this.df.addColumn(col, cardDummies.column(col));
      });
      
      this.df = this.df.drop({ columns: ['Card Type'] });
      console.log(`    Card Type one-hot encoded (${cardDummies.columns.length} columns)`);
    }

    console.log(` Shape after encoding: ${this.df.shape[0]} rows × ${this.df.shape[1]} columns`);
  }

  /**
   * Create new features from existing ones (Feature Engineering)
   */
  createNewFeatures(): void {
    console.log('\n Creating new features ');
    
    // 1. Tenure to Age Ratio (how long customer has been with bank relative to age)
    if (this.df.columns.includes('Tenure') && this.df.columns.includes('Age')) {
      const tenureAgeRatio = this.df.column('Tenure').div(this.df.column('Age'));
      this.df.addColumn('Tenure_Age_Ratio', tenureAgeRatio);
      console.log('   ✓ Created: Tenure_Age_Ratio');
    }

    // 2. Balance per Product (average balance per product)
    if (this.df.columns.includes('Balance') && this.df.columns.includes('NumOfProducts')) {
      const balancePerProduct = this.df.column('Balance').div(this.df.column('NumOfProducts').add(1)); // +1 to avoid division by zero
      this.df.addColumn('Balance_Per_Product', balancePerProduct);
      console.log('   ✓ Created: Balance_Per_Product');
    }

    // 3. Credit Score Category (categorize credit scores)
    if (this.df.columns.includes('CreditScore')) {
      const creditScoreCategory = this.df.column('CreditScore').map((score: any) => {
        if (score >= 800) return 4; // Excellent
        if (score >= 700) return 3; // Good
        if (score >= 600) return 2; // Fair
        return 1; // Poor
      });
      this.df.addColumn('CreditScore_Category', creditScoreCategory);
      console.log(' Created: CreditScore_Category');
    }

    // 4. Age Group
    if (this.df.columns.includes('Age')) {
      const ageGroup = this.df.column('Age').map((age: any) => {
        if (age < 30) return 1; // Young
        if (age < 50) return 2; // Middle-aged
        return 3; // Senior
      });
      this.df.addColumn('Age_Group', ageGroup);
      console.log('Created: Age_Group');
    }

    console.log(` Total features now: ${this.df.shape[1]} columns`);
  }

  /**
   * Normalize numerical features to 0-1 range
   * Important for neural network training
   */
  normalizeFeatures(): void {
    console.log('\n Normalizing numerical features ');

    
    // Columns to normalize (exclude target and already encoded binary columns)
    const columnsToNormalize = [
      'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
      'EstimatedSalary', 'Tenure_Age_Ratio', 'Balance_Per_Product'
    ];

    columnsToNormalize.forEach(col => {
      if (this.df.columns.includes(col)) {
        const column = this.df.column(col);
        const min = column.min();
        const values = column.values as number[];
      
      // Calculate mean
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      
      // Calculate standard deviation
      const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
      const std = Math.sqrt(variance);
      
      if (std > 0) {
        // Standardize: (x - mean) / std
        const standardized = column.sub(mean).div(std);
        
        // Replace original column
        this.df = this.df.drop({ columns: [col] });
        this.df.addColumn(col, standardized);
        console.log(`    Standardized: ${col} (mean: ${mean.toFixed(2)}, std: ${std.toFixed(2)})`);
      }
      }
    });

    console.log(' Normalization complete');
  }

  /**
   * Get the processed DataFrame
   */
  getProcessedData(): dfd.DataFrame {
    return this.df;
  }

  /**
   * Display summary of processed data
   */
  showSummary(): void {
    console.log('\n Processed Data Summary:');
    console.log('═'.repeat(50));
    console.log(`Total Rows: ${this.df.shape[0]}`);
    console.log(`Total Features: ${this.df.shape[1]}`);
    console.log('\nColumns:');
    this.df.columns.forEach((col: string, idx: number) => {
      console.log(`  ${idx + 1}. ${col}`);
    });
    
    // Show first few rows
    console.log('\n First 3 rows of processed data:');
    this.df.head(3).print();
  }

  /**
   * Save processed data to CSV
   */
  async saveProcessedData(outputPath?: string): Promise<void> {
    const savePath = outputPath || path.join(
      process.env.PROCESSED_DATA_PATH || './data/processed',
      'processed_churn_data.csv'
    );

    // Ensure directory exists
    const dir = path.dirname(savePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    await dfd.toCSV(this.df, { filePath: savePath });
    console.log(`\n Processed data saved to: ${savePath}`);
  }
}

/**
 * Main preprocessing pipeline
 */
async function main() {
  console.log(' Bank Customer Churn - Data Preprocessing Pipeline');
  console.log('═'.repeat(50));

  try {
    // 1. Load raw data
    const rawDataPath = process.env.RAW_DATA_PATH || './data/raw/Customer-Churn-Records.csv';
    console.log(`\n Loading data from: ${rawDataPath}`);
    const df = await dfd.readCSV(rawDataPath);
    console.log(` Loaded: ${df.shape[0]} rows × ${df.shape[1]} columns`);

    // 2. Initialize preprocessor
    let preprocessor = new DataPreprocessor(df);

    // 3. Run preprocessing steps
    preprocessor.checkMissingValues();
    preprocessor.dropUnnecessaryColumns();
    preprocessor.encodeCategoricalVariables();
    preprocessor.createNewFeatures();
    preprocessor.normalizeFeatures();

    // 4. DEBUG: Show ALL available columns BEFORE filtering
    const processedDF = preprocessor.getProcessedData();
    console.log('\n DEBUG: All available columns BEFORE filtering:');
    processedDF.columns.forEach((col: string, idx: number) => {
      console.log(`  ${idx + 1}. ${col}`);
    });

    // 5. Keep only normalized columns and essential features
    console.log('\n Selecting final features...');
    
    const columnsToKeep = [
    // Normalized numerical features (now using original names)
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
    'Tenure_Age_Ratio', 'Balance_Per_Product',
    
    // Binary features
    'HasCrCard', 'IsActiveMember', 
    
    // Engineered categorical features
    'CreditScore_Category', 'Age_Group', 'Gender_Encoded',
    
    // Geography one-hot
    'Geography_France', 'Geography_Germany', 'Geography_Spain',
    
    // Card Type one-hot
    'CardType_DIAMOND', 'CardType_GOLD', 'CardType_PLATINUM', 'CardType_SILVER',
    
    // Target
    'Exited'
  ];

    // Filter columns that actually exist
    const availableColumns = columnsToKeep.filter(col => processedDF.columns.includes(col));
    console.log(`   Requested ${columnsToKeep.length} columns`);
    console.log(`   Found ${availableColumns.length} columns`);
    console.log('\n   Available columns:');
    availableColumns.forEach(col => console.log(`     - ${col}`));
    
    const finalDF = processedDF.loc({ columns: availableColumns });
    
    // Create new preprocessor with final data
    preprocessor = new DataPreprocessor(finalDF);

    // 6. Show summary of final data
    preprocessor.showSummary();

    // 7. Save processed data
    await preprocessor.saveProcessedData();

    console.log('\n Preprocessing complete!');
    console.log('\nNext steps:');
    console.log('  1. Run data splitting: npm run split');
    console.log('  2. Train the model: npm run train');

  } catch (error) {
    console.error(' Error in preprocessing:', error);
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main();
}
