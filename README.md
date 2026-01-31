# Multiclass Classification of Flower Species Using CNN

## Project Overview

This graduation project implements a Convolutional Neural Network (CNN) for classifying 5 different flower species using TensorFlow. The implementation uses pure TensorFlow operations (without Keras high-level API) to demonstrate understanding of the underlying neural network mechanics.

**Flower Classes:**
- Lilly
- Lotus
- Orchid
- Sunflower
- Tulip

---

## Technical Specifications

### Model Architecture

The CNN follows a classic architecture with two convolutional blocks followed by fully connected layers:

```
Input (128×128×3)
    ↓
[Conv Layer 1] → 3×3 kernel, stride=1, 32 filters, padding='SAME'
    ↓
[ReLU Activation]
    ↓
[Max Pooling] → 2×2, stride=2 → Output: 64×64×32
    ↓
[Conv Layer 2] → 3×3 kernel, stride=1, 64 filters, padding='SAME'
    ↓
[ReLU Activation]
    ↓
[Max Pooling] → 2×2, stride=2 → Output: 32×32×64
    ↓
[Flatten] → 32×32×64 = 65,536 units
    ↓
[Fully Connected] → 128 units + ReLU + Dropout (0.5 during training)
    ↓
[Output Layer] → 5 units + Softmax
```

### Layer Details

| Layer | Configuration | Output Shape |
|-------|--------------|--------------|
| Input | 128×128 RGB images | (batch, 128, 128, 3) |
| Conv1 | 32 filters, 3×3 kernel, stride=1, ReLU | (batch, 128, 128, 32) |
| Pool1 | 2×2 max pooling, stride=2 | (batch, 64, 64, 32) |
| Conv2 | 64 filters, 3×3 kernel, stride=1, ReLU | (batch, 64, 64, 64) |
| Pool2 | 2×2 max pooling, stride=2 | (batch, 32, 32, 64) |
| Flatten | - | (batch, 65536) |
| FC1 | 128 units, ReLU, Dropout=0.5 | (batch, 128) |
| Output | 5 units, Softmax | (batch, 5) |

### Optimizer: Adam (Backpropagation Algorithm)

The project uses the **Adam (Adaptive Moment Estimation)** optimizer for training. Adam was selected because:

- **Adaptive Learning Rates**: Automatically adjusts learning rates for each parameter based on first and second moments of gradients
- **Momentum Integration**: Combines benefits of RMSprop (adaptive learning rates) and Momentum (accelerated gradients)
- **Sparse Gradient Handling**: Performs well with sparse gradients common in image data
- **Minimal Tuning Required**: Works well with default hyperparameters
- **Memory Efficient**: Requires only first-order gradients with little memory overhead

### Loss Function

**Categorical Cross-Entropy** is used, which is the standard loss function for multi-class classification:

```
Loss = -Σ y_true * log(y_pred)
```

This loss function penalizes confident wrong predictions more heavily, encouraging the model to output well-calibrated probabilities.

---

## Project Structure

```
project/
├── flower_images/           # Dataset directory
│   ├── Lilly/              # Lilly flower images (.jpg)
│   ├── Lotus/              # Lotus flower images (.jpg)
│   ├── Orchid/             # Orchid flower images (.jpg)
│   ├── Sunflower/          # Sunflower flower images (.jpg)
│   └── Tulip/              # Tulip flower images (.jpg)
├── main.py                  # Main source code
├── README.md                # This documentation file
└── requirements.txt         # Python dependencies
```

---

## Code Structure (main.py)

The code is organized into several classes and functions:

### Classes

#### 1. `FlowerCNN`
The main CNN model class that builds and manages the neural network.

**Methods:**
- `__init__(input_shape, num_classes, learning_rate)`: Initializes model parameters
- `build_model()`: Constructs the TensorFlow computation graph including:
  - Placeholder definitions for inputs, labels, and training flag
  - Convolutional layers with variable scopes
  - Fully connected layers
  - Loss function and optimizer
  - Accuracy metric
- `_weight_variable(shape, name)`: Creates weight variables with truncated normal initialization (stddev=0.1)
- `_bias_variable(shape, name)`: Creates bias variables initialized to 0.1
- `_print_model_summary()`: Displays the model architecture

#### 2. `FlowerDataLoader`
Handles all data loading and preprocessing operations.

**Methods:**
- `__init__(data_dir, img_size)`: Sets up data directory and image dimensions
- `load_data()`: 
  - Iterates through class folders
  - Loads images using OpenCV
  - Resizes to target dimensions (128×128)
  - Converts BGR to RGB color space
  - Normalizes pixel values to [0, 1] range
  - Encodes labels using sklearn's LabelEncoder
  - One-hot encodes labels for classification
- `split_data(X, y, test_size, val_size)`: Splits data into train/validation/test sets with stratification

### Functions

#### 3. `train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size)`
Executes the training loop:
- Initializes TensorFlow session and variables
- Creates a Saver for model checkpointing
- For each epoch:
  - Shuffles training data
  - Iterates through mini-batches
  - Runs training operation and computes metrics
  - Evaluates on validation set
  - Saves model if validation accuracy improves
- Returns session, training history, and saver

#### 4. `evaluate_model(sess, model, X_test, y_test, label_encoder)`
Evaluates the trained model:
- Computes test loss and accuracy
- Generates predictions for all test samples
- Calculates and displays per-class accuracy
- Returns metrics and predictions

#### 5. `plot_training_history(history)`
Creates visualization of training progress:
- Subplot 1: Training and validation loss curves
- Subplot 2: Training and validation accuracy curves
- Saves as 'training_history.png'

#### 6. `plot_sample_predictions(sess, model, X_test, y_test, label_encoder, num_samples)`
Visualizes model predictions:
- Randomly selects test samples
- Displays images with true labels, predicted labels, and confidence scores
- Color-codes correct (green) vs incorrect (red) predictions
- Saves as 'sample_predictions.png'

#### 7. `main()`
Orchestrates the complete pipeline:
1. Sets configuration parameters
2. Loads and preprocesses data
3. Splits data into train/validation/test sets
4. Builds the CNN model
5. Trains the model
6. Restores best checkpoint
7. Evaluates on test set
8. Generates visualizations
9. Saves results to text file

---

## Data Processing Pipeline

### Image Preprocessing
1. **Loading**: Images are read using OpenCV (`cv2.imread`)
2. **Resizing**: All images are resized to 128×128 pixels
3. **Color Conversion**: BGR (OpenCV default) is converted to RGB
4. **Normalization**: Pixel values are scaled from [0, 255] to [0, 1]

### Data Splitting
The dataset is split using stratified sampling to maintain class distribution:

| Set | Percentage | Purpose |
|-----|------------|---------|
| Training | ~72% | Model learning |
| Validation | ~8% | Hyperparameter tuning, early stopping |
| Test | 20% | Final evaluation |

*Note: The code uses `test_size=0.2` for test split, then `val_size=0.1` from the original data for validation.*

### Label Encoding
- Labels are first encoded to integers using `LabelEncoder`
- Then converted to one-hot vectors for cross-entropy loss computation

---

## Setup Instructions

### 1. Install Dependencies

Create a `requirements.txt` file with:
```
tensorflow==2.15.0
numpy==1.24.3
opencv-python==4.8.1.78
scikit-learn==1.3.2
matplotlib==3.8.2
```

Install using:
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your flower images in the following structure:
```
flower_images/
├── Lilly/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Lotus/
│   └── ...
├── Orchid/
│   └── ...
├── Sunflower/
│   └── ...
└── Tulip/
    └── ...
```

**Requirements:**
- All images must be `.jpg` format
- Folder names must match exactly: `Lilly`, `Lotus`, `Orchid`, `Sunflower`, `Tulip`
- The `flower_images` folder must be in the same directory as `main.py`

### 3. Run the Project

```bash
python main.py
```

---

## Configuration Parameters

These parameters can be modified in the `main()` function:

```python
DATA_DIR = './flower_images'    # Path to dataset
IMG_SIZE = (128, 128)           # Input image dimensions
EPOCHS = 50                     # Number of training epochs
BATCH_SIZE = 32                 # Mini-batch size
LEARNING_RATE = 0.001           # Adam optimizer learning rate
```

---

## Output Files

After running the project, the following files are generated:

| File | Description |
|------|-------------|
| `best_model.ckpt.meta` | Model architecture |
| `best_model.ckpt.index` | Checkpoint index |
| `best_model.ckpt.data-00000-of-00001` | Model weights |
| `training_history.png` | Loss and accuracy curves |
| `sample_predictions.png` | Visual prediction examples |
| `results.txt` | Detailed results summary |

---

## Expected Console Output

```
============================================================
MULTICLASS CLASSIFICATION OF FLOWER SPECIES
Using CNN with TensorFlow
============================================================

Loading data from: ./flower_images
Loading X images from Lilly...
Loading X images from Lotus...
...

Building CNN Model...
============================================================
CNN MODEL ARCHITECTURE SUMMARY
============================================================
Input Shape: (128, 128, 3)

Layer 1 - Convolutional:
  - Kernel: 3x3, Stride: 1, Filters: 32
  - Activation: ReLU
  - Max Pooling: 2x2
...

============================================================
STARTING TRAINING
============================================================
Epoch 1/50 - Train Loss: X.XXXX, Train Acc: X.XXXX - Val Loss: X.XXXX, Val Acc: X.XXXX
  → New best model saved! (Val Acc: X.XXXX)
...

============================================================
EVALUATING MODEL ON TEST SET
============================================================
Test Loss: X.XXXX
Test Accuracy: X.XXXX

Per-Class Accuracy:
  Lilly: X.XXXX
  Lotus: X.XXXX
  Orchid: X.XXXX
  Sunflower: X.XXXX
  Tulip: X.XXXX
```

---

## Technical Implementation Notes

### TensorFlow 1.x Compatibility Mode
The code uses `tf.compat.v1` to:
- Disable eager execution for explicit graph construction
- Use placeholders for input data
- Manage sessions manually
- Provide explicit control over the computation graph

This approach demonstrates understanding of low-level TensorFlow operations rather than relying on Keras abstractions.

### Weight Initialization
- **Weights**: Truncated normal distribution with stddev=0.1
- **Biases**: Constant value of 0.1

### Regularization Techniques
- **Dropout**: 50% dropout rate applied to fully connected layer during training only
- **Model Checkpointing**: Saves best model based on validation accuracy (early stopping mechanism)

### Random Seed Setting
Seeds are set for reproducibility:
```python
np.random.seed(42)
tf.random.set_seed(42)
```

---

## Troubleshooting

### "No module named 'tensorflow'"
```bash
pip install tensorflow==2.15.0
```

### "No module named 'cv2'"
```bash
pip install opencv-python
```

### "Directory not found" or "Warning: Directory X not found!"
- Ensure `flower_images` folder exists in the same directory as `main.py`
- Verify folder names are spelled correctly (case-sensitive)

### Out of Memory Error
Reduce memory usage by modifying in `main()`:
```python
BATCH_SIZE = 16        # Reduce from 32
IMG_SIZE = (64, 64)    # Reduce from (128, 128)
```

### Low Accuracy
- Increase training epochs: `EPOCHS = 100`
- Ensure sufficient training data (recommend 500+ images per class)
- Verify images are properly formatted and not corrupted
- Try adjusting learning rate: `LEARNING_RATE = 0.0001`

### Training is Very Slow
- Install TensorFlow GPU version for CUDA-enabled GPUs
- Reduce image size or batch size
- Ensure no other heavy processes are running

---

## System Requirements

- **Python**: 3.8 - 3.12
- **RAM**: Minimum 8GB recommended
- **Storage**: Depends on dataset size
- **GPU**: Optional (NVIDIA with CUDA support for faster training)

---

## Author

Computer Science Student - Graduation Project

---

## License

This project is for educational purposes as part of a graduation requirement.
