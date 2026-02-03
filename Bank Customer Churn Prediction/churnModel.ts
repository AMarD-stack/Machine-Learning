import * as tf from '@tensorflow/tfjs-node';

/**
 * ChurnModel - Neural Network architecture for predicting customer churn
 * Following ML Lifecycle: Model Selection & Training phase
 */
export class ChurnModel {
  private model: tf.Sequential | null = null;
  private inputShape: number;
  private optimalThreshold: number = 0.5;

  constructor(inputShape: number) {
    this.inputShape = inputShape;
  }

  /**
   * Build the neural network architecture
   */
  buildModel(): tf.Sequential {
    console.log('\n  Building Neural Network Model...');
    console.log('─'.repeat(50));

    this.model = tf.sequential();

    // Input Layer + First Hidden Layer - 128 neurons
    this.model.add(tf.layers.dense({
      inputShape: [this.inputShape],
      units: 128,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      name: 'dense_1'
    }));

    this.model.add(tf.layers.dropout({
      rate: 0.4,
      name: 'dropout_1'
    }));

    // Second Hidden Layer - 64 neurons
    this.model.add(tf.layers.dense({
      units: 64,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      name: 'dense_2'
    }));

    this.model.add(tf.layers.dropout({
      rate: 0.3,
      name: 'dropout_2'
    }));

    // Third Hidden Layer - 32 neurons
    this.model.add(tf.layers.dense({
      units: 32,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      name: 'dense_3'
    }));

    this.model.add(tf.layers.dropout({
      rate: 0.2,
      name: 'dropout_3'
    }));

    // Output Layer - 1 neuron, Sigmoid activation
    this.model.add(tf.layers.dense({
      units: 1,
      activation: 'sigmoid',
      name: 'output'
    }));

    console.log(' Model architecture created');
    console.log('\nModel Summary:');
    this.model.summary();

    return this.model;
  }

  /**
   * Compile the model with optimizer and loss function
   */
  compileModel(learningRate: number = 0.001): void {
    if (!this.model) {
      throw new Error('Model not built. Call buildModel() first.');
    }

    console.log('\n  Compiling model...');
    console.log(`   Learning Rate: ${learningRate}`);

    const optimizer = tf.train.adam(learningRate);

    this.model.compile({
      optimizer: optimizer,
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });

    console.log(' Model compiled');
    console.log('   Optimizer: Adam');
    console.log('   Loss Function: Binary Crossentropy');
    console.log('   Metrics: Accuracy');
  }

  /**
   * Train the model with balanced class weights
   */
  async trainModel(
    trainX: tf.Tensor2D,
    trainY: tf.Tensor2D,
    valX: tf.Tensor2D,
    valY: tf.Tensor2D,
    epochs: number = 100,
    batchSize: number = 32
  ): Promise<tf.History> {
    if (!this.model) {
      throw new Error('Model not built. Call buildModel() first.');
    }

    console.log('\n Starting Training...');
    console.log('═'.repeat(50));

    // Calculate balanced class weights
    const totalSamples = trainY.shape[0];
    const trainYArray = await trainY.array() as number[][];
    const positives = trainYArray.filter(y => y[0] === 1).length;
    const negatives = totalSamples - positives;

    const classWeight = {
      0: 1.0,
      1: negatives / positives  // Balanced weight (should be ~3.9)
    };

    console.log(`Class Distribution:`);
    console.log(`   Negatives (No Churn): ${negatives} (${((negatives/totalSamples)*100).toFixed(1)}%)`);
    console.log(`   Positives (Churn): ${positives} (${((positives/totalSamples)*100).toFixed(1)}%)`);
    console.log(`Class Weights: 0=${classWeight[0].toFixed(2)}, 1=${classWeight[1].toFixed(2)}`);
    console.log(`Epochs: ${epochs}`);
    console.log(`Batch Size: ${batchSize}`);

    // Early stopping callback
    const earlyStoppingCallback = tf.callbacks.earlyStopping({
      monitor: 'val_loss',
      patience: 15
    });

    // Train the model
    const history = await this.model.fit(trainX, trainY, {
      epochs: epochs,
      batchSize: batchSize,
      validationData: [valX, valY],
      callbacks: [earlyStoppingCallback],
      shuffle: true,
      verbose: 1,
      classWeight: classWeight
    });

    console.log('\n Training complete!');

    return history;
  }

  /**
   * Find optimal threshold using validation data
   */
  async findOptimalThreshold(
    valX: tf.Tensor2D,
    valY: tf.Tensor2D
  ): Promise<number> {
    console.log('\n Finding optimal decision threshold...');
    console.log('─'.repeat(50));
    
    const predictions = this.predict(valX);
    const predArray = await predictions.array() as number[][];
    const actualArray = await valY.array() as number[][];
    
    let bestThreshold = 0.5;
    let bestF1 = 0;
    let bestPrecision = 0;
    let bestRecall = 0;
    
    console.log('\nThreshold | Precision | Recall | F1-Score');
    console.log('─'.repeat(50));
    
    // Try thresholds from 0.1 to 0.9
    for (let threshold = 0.1; threshold <= 0.9; threshold += 0.05) {
      let tp = 0, fp = 0, fn = 0;
      
      for (let i = 0; i < predArray.length; i++) {
        const pred = predArray[i][0] > threshold ? 1 : 0;
        const actual = actualArray[i][0];
        
        if (pred === 1 && actual === 1) tp++;
        if (pred === 1 && actual === 0) fp++;
        if (pred === 0 && actual === 1) fn++;
      }
      
      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
      
      if (threshold % 0.1 < 0.051) {
        console.log(`  ${threshold.toFixed(2)}    |   ${(precision*100).toFixed(1)}%   |  ${(recall*100).toFixed(1)}%  |  ${(f1*100).toFixed(1)}%`);
      }
      
      if (f1 > bestF1) {
        bestF1 = f1;
        bestThreshold = threshold;
        bestPrecision = precision;
        bestRecall = recall;
      }
    }
    
    console.log('─'.repeat(50));
    console.log(` Optimal threshold: ${bestThreshold.toFixed(2)}`);
    console.log(`   Best F1-Score: ${(bestF1*100).toFixed(2)}%`);
    console.log(`   Precision: ${(bestPrecision*100).toFixed(2)}%`);
    console.log(`   Recall: ${(bestRecall*100).toFixed(2)}%`);
    
    predictions.dispose();
    
    this.optimalThreshold = bestThreshold;
    return bestThreshold;
  }

  /**
   * Evaluate model on test data
   */
  async evaluateModel(testX: tf.Tensor2D, testY: tf.Tensor2D): Promise<{
    loss: number;
    accuracy: number;
  }> {
    if (!this.model) {
      throw new Error('Model not built. Call buildModel() first.');
    }

    console.log('\n Evaluating model on test set...');
    
    const evaluation = await this.model.evaluate(testX, testY) as tf.Scalar[];
    const loss = await evaluation[0].data();
    const accuracy = await evaluation[1].data();

    console.log('─'.repeat(50));
    console.log(`Test Loss: ${loss[0].toFixed(4)}`);
    console.log(`Test Accuracy: ${(accuracy[0] * 100).toFixed(2)}%`);
    console.log('─'.repeat(50));

    evaluation.forEach(tensor => tensor.dispose());

    return {
      loss: loss[0],
      accuracy: accuracy[0]
    };
  }

  /**
   * Make predictions on new data
   */
  predict(data: tf.Tensor2D): tf.Tensor {
    if (!this.model) {
      throw new Error('Model not built. Call buildModel() first.');
    }

    return this.model.predict(data) as tf.Tensor;
  }

  /**
   * Save the trained model
   */
  async saveModel(savePath: string = 'file://./models/churn-model'): Promise<void> {
    if (!this.model) {
      throw new Error('Model not built. Call buildModel() first.');
    }

    console.log(`\n Saving model to: ${savePath}`);
    await this.model.save(savePath);
    console.log(' Model saved successfully');
    console.log(`   Optimal threshold: ${this.optimalThreshold.toFixed(2)}`);
  }

  /**
   * Load a saved model
   */
  async loadModel(loadPath: string = 'file://./models/churn-model/model.json'): Promise<void> {
    console.log(`\n Loading model from: ${loadPath}`);
    this.model = await tf.loadLayersModel(loadPath) as tf.Sequential;
    console.log(' Model loaded successfully');
  }

  /**
   * Get the model
   */
  getModel(): tf.Sequential {
    if (!this.model) {
      throw new Error('Model not built. Call buildModel() first.');
    }
    return this.model;
  }

  /**
   * Get optimal threshold
   */
  getOptimalThreshold(): number {
    return this.optimalThreshold;
  }

  /**
   * Calculate metrics with custom threshold
   */
  async calculateMetrics(
    predictions: tf.Tensor,
    actual: tf.Tensor2D,
    threshold: number = 0.5
  ): Promise<{
    precision: number;
    recall: number;
    f1Score: number;
  }> {
    console.log(`\n Calculating metrics (threshold: ${threshold.toFixed(2)})...`);

    const predValues = await predictions.array() as number[][];
    const actualValues = await actual.array() as number[][];

    let tp = 0, fp = 0, fn = 0, tn = 0;

    for (let i = 0; i < predValues.length; i++) {
      const pred = predValues[i][0] > threshold ? 1 : 0;
      const act = actualValues[i][0];

      if (pred === 1 && act === 1) tp++;
      if (pred === 1 && act === 0) fp++;
      if (pred === 0 && act === 1) fn++;
      if (pred === 0 && act === 0) tn++;
    }

    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1Score = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

    console.log('─'.repeat(50));
    console.log(`Confusion Matrix:`);
    console.log(`  TP: ${tp} | FP: ${fp}`);
    console.log(`  FN: ${fn} | TN: ${tn}`);
    console.log(`Precision: ${(precision * 100).toFixed(2)}%`);
    console.log(`Recall: ${(recall * 100).toFixed(2)}%`);
    console.log(`F1-Score: ${(f1Score * 100).toFixed(2)}%`);
    console.log('─'.repeat(50));

    return { precision, recall, f1Score };
  }
}

async function example() {
  console.log(' Bank Customer Churn - Model Architecture Demo');
  console.log('═'.repeat(50));

  const inputShape = 17;
  const model = new ChurnModel(inputShape);

  model.buildModel();
  model.compileModel(0.001);

  console.log('\n✨ Model is ready for training!');
  console.log('\nTo train the model, run:');
  console.log('  npx ts-node src/models/trainer.ts');
}

if (require.main === module) {
  example();
}