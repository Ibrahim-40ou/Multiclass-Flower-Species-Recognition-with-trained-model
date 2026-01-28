"""
Multiclass Classification of Flower Species using CNN
Student Graduation Project
Using TensorFlow (without Keras)
"""

import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

# Disable eager execution for TensorFlow 2.x compatibility with TF 1.x style code
tf.compat.v1.disable_eager_execution()

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class FlowerCNN:
    """
    Convolutional Neural Network for Flower Species Classification
    Architecture:
    - Input Layer
    - Conv Layer 1 (3x3 kernel, stride=1) + ReLU + MaxPooling
    - Conv Layer 2 (3x3 kernel, stride=1) + ReLU + MaxPooling
    - Fully Connected Layer + ReLU
    - Output Layer + Softmax
    """
    
    def __init__(self, input_shape=(128, 128, 3), num_classes=5, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.build_model()
        
    def build_model(self):
        """Build the CNN architecture using TensorFlow"""
        print("Building CNN Model...")
        
        # Define input placeholder
        self.X = tf.compat.v1.placeholder(tf.float32, 
                                          shape=[None, *self.input_shape], 
                                          name='input')
        self.Y = tf.compat.v1.placeholder(tf.float32, 
                                          shape=[None, self.num_classes], 
                                          name='labels')
        self.is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')
        
        # Convolutional Layer 1: 3x3 kernel, stride=1
        with tf.compat.v1.variable_scope('conv1'):
            self.W_conv1 = self._weight_variable([3, 3, 3, 32], name='weights')
            self.b_conv1 = self._bias_variable([32], name='bias')
            self.h_conv1 = tf.nn.conv2d(self.X, self.W_conv1, 
                                       strides=[1, 1, 1, 1], 
                                       padding='SAME')
            self.h_conv1 = tf.nn.bias_add(self.h_conv1, self.b_conv1)
            # ReLU activation
            self.h_conv1_relu = tf.nn.relu(self.h_conv1)
            # Max pooling 2x2
            self.h_pool1 = tf.nn.max_pool2d(self.h_conv1_relu, 
                                           ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1], 
                                           padding='SAME')
        
        # Convolutional Layer 2: 3x3 kernel, stride=1
        with tf.compat.v1.variable_scope('conv2'):
            self.W_conv2 = self._weight_variable([3, 3, 32, 64], name='weights')
            self.b_conv2 = self._bias_variable([64], name='bias')
            self.h_conv2 = tf.nn.conv2d(self.h_pool1, self.W_conv2,
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')
            self.h_conv2 = tf.nn.bias_add(self.h_conv2, self.b_conv2)
            # ReLU activation
            self.h_conv2_relu = tf.nn.relu(self.h_conv2)
            # Max pooling 2x2
            self.h_pool2 = tf.nn.max_pool2d(self.h_conv2_relu,
                                           ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1],
                                           padding='SAME')
        
        # Flatten the feature maps
        pool2_shape = self.h_pool2.get_shape().as_list()
        flatten_size = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, flatten_size])
        
        # Fully Connected Layer (Hidden Layer)
        with tf.compat.v1.variable_scope('fc1'):
            self.W_fc1 = self._weight_variable([flatten_size, 128], name='weights')
            self.b_fc1 = self._bias_variable([128], name='bias')
            self.h_fc1 = tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1
            # ReLU activation
            self.h_fc1_relu = tf.nn.relu(self.h_fc1)
            # Dropout for regularization
            # Use conditional to apply dropout only during training
            self.keep_prob = tf.cond(self.is_training, 
                                    lambda: tf.constant(0.5), 
                                    lambda: tf.constant(1.0))
            self.h_fc1_drop = tf.nn.dropout(self.h_fc1_relu, rate=1 - self.keep_prob)
        
        # Output Layer
        with tf.compat.v1.variable_scope('output'):
            self.W_fc2 = self._weight_variable([128, self.num_classes], name='weights')
            self.b_fc2 = self._bias_variable([self.num_classes], name='bias')
            self.logits = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
            # Softmax activation for probabilities
            self.predictions = tf.nn.softmax(self.logits)
        
        # Loss function: Cross-entropy
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, 
                                                    logits=self.logits))
        
        # Optimizer: Adam (adaptive learning rate, suitable for this application)
        # Adam is chosen as it's well-suited for image classification with adaptive learning rates
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        
        # Accuracy metric
        correct_prediction = tf.equal(tf.argmax(self.predictions, 1), 
                                     tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        print("Model built successfully!")
        self._print_model_summary()
    
    def _weight_variable(self, shape, name):
        """Initialize weights with Xavier initialization"""
        initial = tf.compat.v1.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)
    
    def _bias_variable(self, shape, name):
        """Initialize biases with small constant values"""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)
    
    def _print_model_summary(self):
        """Print model architecture summary"""
        print("\n" + "="*60)
        print("CNN MODEL ARCHITECTURE SUMMARY")
        print("="*60)
        print(f"Input Shape: {self.input_shape}")
        print(f"\nLayer 1 - Convolutional:")
        print(f"  - Kernel: 3x3, Stride: 1, Filters: 32")
        print(f"  - Activation: ReLU")
        print(f"  - Max Pooling: 2x2")
        print(f"\nLayer 2 - Convolutional:")
        print(f"  - Kernel: 3x3, Stride: 1, Filters: 64")
        print(f"  - Activation: ReLU")
        print(f"  - Max Pooling: 2x2")
        print(f"\nLayer 3 - Fully Connected (Hidden):")
        print(f"  - Units: 128")
        print(f"  - Activation: ReLU")
        print(f"  - Dropout: 0.5")
        print(f"\nLayer 4 - Output:")
        print(f"  - Units: {self.num_classes}")
        print(f"  - Activation: Softmax")
        print(f"\nOptimizer: Adam (learning_rate={self.learning_rate})")
        print(f"Loss Function: Categorical Cross-Entropy")
        print("="*60 + "\n")


class FlowerDataLoader:
    """Load and preprocess flower images dataset"""
    
    def __init__(self, data_dir, img_size=(128, 128)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.class_names = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']
        
    def load_data(self):
        """Load images and labels from directory structure"""
        print(f"\nLoading data from: {self.data_dir}")
        images = []
        labels = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found!")
                continue
            
            image_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
            print(f"Loading {len(image_files)} images from {class_name}...")
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Resize image
                    img = cv2.resize(img, self.img_size)
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Normalize pixel values to [0, 1]
                    img = img.astype(np.float32) / 255.0
                    
                    images.append(img)
                    labels.append(class_name)
        
        print(f"\nTotal images loaded: {len(images)}")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # One-hot encode labels
        y_onehot = np.eye(len(self.class_names))[y_encoded]
        
        return X, y_onehot, label_encoder
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        # First split: train+val and test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)
        
        # Second split: train and validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_ratio, random_state=42, 
            stratify=y_trainval)
        
        print(f"\nData split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Test samples: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=50, batch_size=32):
    """Train the CNN model"""
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    num_batches = len(X_train) // batch_size
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Initialize TensorFlow session
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # Create saver for model checkpoints
    saver = tf.compat.v1.train.Saver()
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Training
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            
            batch_X = X_train_shuffled[start_idx:end_idx]
            batch_y = y_train_shuffled[start_idx:end_idx]
            
            _, loss, acc = sess.run(
                [model.train_op, model.loss, model.accuracy],
                feed_dict={
                    model.X: batch_X,
                    model.Y: batch_y,
                    model.is_training: True
                }
            )
            
            epoch_loss += loss
            epoch_acc += acc
        
        # Calculate average training metrics
        avg_train_loss = epoch_loss / num_batches
        avg_train_acc = epoch_acc / num_batches
        
        # Validation
        val_loss, val_acc = sess.run(
            [model.loss, model.accuracy],
            feed_dict={
                model.X: X_val,
                model.Y: y_val,
                model.is_training: False
            }
        )
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            saver.save(sess, './best_model.ckpt')
            print(f"  → New best model saved! (Val Acc: {val_acc:.4f})")
    
    print("\n" + "="*60)
    print(f"TRAINING COMPLETED - Best Validation Accuracy: {best_val_acc:.4f}")
    print("="*60)
    
    return sess, history, saver


def evaluate_model(sess, model, X_test, y_test, label_encoder):
    """Evaluate the model on test data"""
    print("\n" + "="*60)
    print("EVALUATING MODEL ON TEST SET")
    print("="*60)
    
    test_loss, test_acc = sess.run(
        [model.loss, model.accuracy],
        feed_dict={
            model.X: X_test,
            model.Y: y_test,
            model.is_training: False
        }
    )
    
    # Get predictions
    predictions = sess.run(
        model.predictions,
        feed_dict={
            model.X: X_test,
            model.is_training: False
        }
    )
    
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(label_encoder.classes_):
        class_mask = true_classes == i
        class_acc = np.mean(pred_classes[class_mask] == true_classes[class_mask])
        print(f"  {class_name}: {class_acc:.4f}")
    
    print("="*60)
    
    return test_loss, test_acc, predictions


def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\nTraining history plot saved as 'training_history.png'")
    plt.show()


def plot_sample_predictions(sess, model, X_test, y_test, label_encoder, num_samples=10):
    """Plot sample predictions"""
    # Get random samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    sample_images = X_test[indices]
    sample_labels = y_test[indices]
    
    # Get predictions
    predictions = sess.run(
        model.predictions,
        feed_dict={
            model.X: sample_images,
            model.is_training: False
        }
    )
    
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(sample_labels, axis=1)
    
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        axes[i].imshow(sample_images[i])
        pred_label = label_encoder.classes_[pred_classes[i]]
        true_label = label_encoder.classes_[true_classes[i]]
        confidence = predictions[i][pred_classes[i]] * 100
        
        color = 'green' if pred_classes[i] == true_classes[i] else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\n({confidence:.1f}%)',
                         fontsize=9, color=color, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
    print("Sample predictions plot saved as 'sample_predictions.png'")
    plt.show()


def main():
    """Main function to run the complete pipeline"""
    print("\n" + "="*60)
    print("MULTICLASS CLASSIFICATION OF FLOWER SPECIES")
    print("Using CNN with TensorFlow")
    print("="*60)
    
    # Configuration
    DATA_DIR = './flower_images'
    IMG_SIZE = (128, 128)
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    # Load data
    data_loader = FlowerDataLoader(DATA_DIR, IMG_SIZE)
    X, y, label_encoder = data_loader.load_data()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(X, y)
    
    # Build model
    model = FlowerCNN(input_shape=(*IMG_SIZE, 3), 
                     num_classes=len(label_encoder.classes_),
                     learning_rate=LEARNING_RATE)
    
    # Train model
    sess, history, saver = train_model(model, X_train, y_train, X_val, y_val,
                                      epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Load best model
    saver.restore(sess, './best_model.ckpt')
    print("\nBest model restored for final evaluation.")
    
    # Evaluate model
    test_loss, test_acc, predictions = evaluate_model(sess, model, X_test, y_test, 
                                                      label_encoder)
    
    # Plot results
    plot_training_history(history)
    plot_sample_predictions(sess, model, X_test, y_test, label_encoder)
    
    # Save final results
    results_file = 'results.txt'
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FLOWER CLASSIFICATION PROJECT RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Model Architecture:\n")
        f.write(f"  - Input: {IMG_SIZE} RGB images\n")
        f.write(f"  - Conv Layer 1: 3x3 kernel, stride=1, 32 filters, ReLU\n")
        f.write(f"  - Conv Layer 2: 3x3 kernel, stride=1, 64 filters, ReLU\n")
        f.write(f"  - FC Layer: 128 units, ReLU\n")
        f.write(f"  - Output: {len(label_encoder.classes_)} classes, Softmax\n")
        f.write(f"  - Optimizer: Adam (lr={LEARNING_RATE})\n\n")
        f.write(f"Training Configuration:\n")
        f.write(f"  - Epochs: {EPOCHS}\n")
        f.write(f"  - Batch Size: {BATCH_SIZE}\n")
        f.write(f"  - Train Samples: {len(X_train)}\n")
        f.write(f"  - Val Samples: {len(X_val)}\n")
        f.write(f"  - Test Samples: {len(X_test)}\n\n")
        f.write(f"Final Results:\n")
        f.write(f"  - Test Loss: {test_loss:.4f}\n")
        f.write(f"  - Test Accuracy: {test_acc:.4f}\n")
        f.write("="*60 + "\n")
    
    print(f"\nResults saved to '{results_file}'")
    print("\nProject completed successfully!")
    
    sess.close()


if __name__ == "__main__":
    main()