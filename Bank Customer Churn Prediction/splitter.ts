import * as dfd from 'danfojs-node';
import * as tf from '@tensorflow/tfjs-node';
import * as path from 'path';
import * as fs from 'fs';
import { config } from 'dotenv';

config();

/**
 * DataSplitter - Splits preprocessed data into train, validation, and test sets
 * Following ML Lifecycle: Data Preparation phase
 */
export class DataSplitter {
  private df: dfd.DataFrame;
  private targetColumn: string;

  constructor(df: dfd.DataFrame, targetColumn: string = 'Exited') {
    this.df = df;
    this.targetColumn = targetColumn;
  }

  /**
   * Shuffle the dataframe to ensure random distribution
   */
  shuffleData(): void {
    console.log(' Shuffling data...');
    
    // Get indices and shuffle them
    const indices = Array.from({ length: this.df.shape[0] }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    
    // Reindex dataframe with shuffled indices
    this.df = this.df.iloc({ rows: indices });
    console.log(' Data shuffled');
  }

  /**
   * Split data into features (X) and target (y)
   */
  separateFeaturesAndTarget(): { X: dfd.DataFrame; y: dfd.Series } {
    console.log('\n Separating features and target...');
    
    // Check if target column exists
    if (!this.df.columns.includes(this.targetColumn)) {
      throw new Error(`Target column '${this.targetColumn}' not found in dataset`);
    }

    // Separate target
    const y = this.df.column(this.targetColumn);
    
    // Get all feature columns (exclude target)
    const X = this.df.drop({ columns: [this.targetColumn] });
    
    console.log(`   Features (X): ${X.shape[1]} columns`);
    console.log(`   Target (y): ${this.targetColumn}`);
    console.log(`   Total samples: ${X.shape[0]}`);
    
    // Check class distribution
    const churnCount = y.sum();
    const retainedCount = y.shape[0] - churnCount;
    const churnRate = (churnCount / y.shape[0] * 100).toFixed(2);
    
    console.log(`\n   Class Distribution:`);
    console.log(`   - Retained (0): ${retainedCount} (${(100 - parseFloat(churnRate)).toFixed(2)}%)`);
    console.log(`   - Churned (1): ${churnCount} (${churnRate}%)`);
    
    return { X, y };
  }

  /**
   * Split data into train, validation, and test sets
   * Default split: 70% train, 15% validation, 15% test
   */
  splitData(trainRatio: number = 0.70, valRatio: number = 0.15): {
    trainX: number[][];
    trainY: number[];
    valX: number[][];
    valY: number[];
    testX: number[][];
    testY: number[];
  } {
    console.log('\n  Splitting data...');
    console.log(`   Train: ${(trainRatio * 100).toFixed(0)}%`);
    console.log(`   Validation: ${(valRatio * 100).toFixed(0)}%`);
    console.log(`   Test: ${((1 - trainRatio - valRatio) * 100).toFixed(0)}%`);

    // Separate features and target
    const { X, y } = this.separateFeaturesAndTarget();
    
    const totalSamples = X.shape[0];
    const trainSize = Math.floor(totalSamples * trainRatio);
    const valSize = Math.floor(totalSamples * valRatio);
    
    // Convert to arrays
    const XArray = X.values as number[][];
    const yArray = y.values as number[];
    
    // Split indices
    const trainX = XArray.slice(0, trainSize);
    const trainY = yArray.slice(0, trainSize);
    
    const valX = XArray.slice(trainSize, trainSize + valSize);
    const valY = yArray.slice(trainSize, trainSize + valSize);
    
    const testX = XArray.slice(trainSize + valSize);
    const testY = yArray.slice(trainSize + valSize);
    
    console.log('\n Split complete:');
    console.log(`   Training set: ${trainX.length} samples`);
    console.log(`   Validation set: ${valX.length} samples`);
    console.log(`   Test set: ${testX.length} samples`);
    
    return { trainX, trainY, valX, valY, testX, testY };
  }

  /**
   * Convert split data to TensorFlow tensors
   */
  convertToTensors(data: {
    trainX: number[][];
    trainY: number[];
    valX: number[][];
    valY: number[];
    testX: number[][];
    testY: number[];
  }): {
    trainXTensor: tf.Tensor2D;
    trainYTensor: tf.Tensor2D;
    valXTensor: tf.Tensor2D;
    valYTensor: tf.Tensor2D;
    testXTensor: tf.Tensor2D;
    testYTensor: tf.Tensor2D;
  } {
    console.log('\n Converting to TensorFlow tensors...');
    
    const trainXTensor = tf.tensor2d(data.trainX);
    const trainYTensor = tf.tensor2d(data.trainY, [data.trainY.length, 1]);
    
    const valXTensor = tf.tensor2d(data.valX);
    const valYTensor = tf.tensor2d(data.valY, [data.valY.length, 1]);
    
    const testXTensor = tf.tensor2d(data.testX);
    const testYTensor = tf.tensor2d(data.testY, [data.testY.length, 1]);
    
    console.log(` Tensors created:`);
    console.log(`   Train X: ${trainXTensor.shape}`);
    console.log(`   Train Y: ${trainYTensor.shape}`);
    console.log(`   Val X: ${valXTensor.shape}`);
    console.log(`   Val Y: ${valYTensor.shape}`);
    console.log(`   Test X: ${testXTensor.shape}`);
    console.log(`   Test Y: ${testYTensor.shape}`);
    
    return {
      trainXTensor,
      trainYTensor,
      valXTensor,
      valYTensor,
      testXTensor,
      testYTensor
    };
  }

  /**
   * Save split data to files for later use
   */
  async saveSplitData(data: {
    trainX: number[][];
    trainY: number[];
    valX: number[][];
    valY: number[];
    testX: number[][];
    testY: number[];
  }): Promise<void> {
    console.log('\n Saving split data...');
    
    const splitsDir = path.join(process.cwd(), 'data', 'splits');
    if (!fs.existsSync(splitsDir)) {
      fs.mkdirSync(splitsDir, { recursive: true });
    }

    // Save as JSON for easy loading
    const splits = {
      train: { X: data.trainX, y: data.trainY },
      validation: { X: data.valX, y: data.valY },
      test: { X: data.testX, y: data.testY },
      metadata: {
        numFeatures: data.trainX[0].length,
        trainSize: data.trainX.length,
        valSize: data.valX.length,
        testSize: data.testX.length,
        createdAt: new Date().toISOString()
      }
    };

    const filePath = path.join(splitsDir, 'data_splits.json');
    fs.writeFileSync(filePath, JSON.stringify(splits, null, 2));
    
    console.log(` Split data saved to: ${filePath}`);
  }

  /**
   * Load previously saved split data
   */
  static loadSplitData(filePath?: string): {
    trainX: number[][];
    trainY: number[];
    valX: number[][];
    valY: number[];
    testX: number[][];
    testY: number[];
  } {
    const loadPath = filePath || path.join(process.cwd(), 'data', 'splits', 'data_splits.json');
    
    if (!fs.existsSync(loadPath)) {
      throw new Error(`Split data not found at: ${loadPath}`);
    }

    const data = JSON.parse(fs.readFileSync(loadPath, 'utf-8'));
    
    return {
      trainX: data.train.X,
      trainY: data.train.y,
      valX: data.validation.X,
      valY: data.validation.y,
      testX: data.test.X,
      testY: data.test.y
    };
  }
}

/**
 * Main function to run the splitting pipeline
 */
async function main() {
  console.log(' Bank Customer Churn - Data Splitting Pipeline');
  console.log('═'.repeat(50));

  try {
    // 1. Load preprocessed data
    const processedDataPath = path.join(
      process.env.PROCESSED_DATA_PATH || './data/processed',
      'processed_churn_data.csv'
    );
    
    console.log(`\n Loading preprocessed data from: ${processedDataPath}`);
    const df = await dfd.readCSV(processedDataPath);
    console.log(` Loaded: ${df.shape[0]} rows × ${df.shape[1]} columns`);

    // 2. Initialize splitter
    const splitter = new DataSplitter(df, 'Exited');

    // 3. Shuffle data
    splitter.shuffleData();

    // 4. Split data
    const splitData = splitter.splitData(0.70, 0.15);

    // 5. Save splits
    await splitter.saveSplitData(splitData);

    // 6. Show tensor conversion example
    const tensors = splitter.convertToTensors(splitData);
    
    // Clean up tensors
    Object.values(tensors).forEach(tensor => tensor.dispose());

    console.log('\n Data splitting complete!');
    console.log('\nNext steps:');
    console.log('  1. Build the model: Create model architecture');
    console.log('  2. Train the model: npm run train');

  } catch (error) {
    console.error(' Error in data splitting:', error);
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main();
}