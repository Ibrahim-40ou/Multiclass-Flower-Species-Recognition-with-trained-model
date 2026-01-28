"""
Quick Test Script - Test your model on any image
Usage: python quick_test.py path/to/your/image.jpg
"""

import tensorflow as tf
import numpy as np
import cv2
import sys
import os

# Disable eager execution
tf.compat.v1.disable_eager_execution()

def predict_flower(image_path):
    """Quick prediction on a single image"""
    
    class_names = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']
    
    print(f"\nLoading model and predicting: {image_path}")
    
    # Read and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not read image from {image_path}")
        return
    
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img, axis=0)
    
    # Build model
    X = tf.compat.v1.placeholder(tf.float32, shape=[None, 128, 128, 3])
    
    # Conv1
    with tf.compat.v1.variable_scope('conv1'):
        W1 = tf.Variable(tf.compat.v1.truncated_normal([3, 3, 3, 32], stddev=0.1), name='weights')
        b1 = tf.Variable(tf.constant(0.1, shape=[32]), name='bias')
        h1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, W1, [1,1,1,1], 'SAME'), b1))
        p1 = tf.nn.max_pool2d(h1, [1,2,2,1], [1,2,2,1], 'SAME')
    
    # Conv2
    with tf.compat.v1.variable_scope('conv2'):
        W2 = tf.Variable(tf.compat.v1.truncated_normal([3, 3, 32, 64], stddev=0.1), name='weights')
        b2 = tf.Variable(tf.constant(0.1, shape=[64]), name='bias')
        h2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(p1, W2, [1,1,1,1], 'SAME'), b2))
        p2 = tf.nn.max_pool2d(h2, [1,2,2,1], [1,2,2,1], 'SAME')
    
    # FC
    flat = tf.reshape(p2, [-1, 32*32*64])
    with tf.compat.v1.variable_scope('fc1'):
        W3 = tf.Variable(tf.compat.v1.truncated_normal([32*32*64, 128], stddev=0.1), name='weights')
        b3 = tf.Variable(tf.constant(0.1, shape=[128]), name='bias')
        h3 = tf.nn.relu(tf.matmul(flat, W3) + b3)
    
    # Output
    with tf.compat.v1.variable_scope('output'):
        W4 = tf.Variable(tf.compat.v1.truncated_normal([128, 5], stddev=0.1), name='weights')
        b4 = tf.Variable(tf.constant(0.1, shape=[5]), name='bias')
        predictions = tf.nn.softmax(tf.matmul(h3, W4) + b4)
    
    # Load and predict
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, './best_model.ckpt')
    
    probs = sess.run(predictions, feed_dict={X: img_batch})[0]
    
    # Results
    pred_idx = np.argmax(probs)
    print("\n" + "="*50)
    print(f"PREDICTION: {class_names[pred_idx]}")
    print(f"CONFIDENCE: {probs[pred_idx]*100:.2f}%")
    print("="*50)
    print("\nAll probabilities:")
    for i, name in enumerate(class_names):
        bar = "█" * int(probs[i] * 50)
        print(f"  {name:12s} {probs[i]*100:6.2f}% {bar}")
    print("="*50)
    
    sess.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_flower(sys.argv[1])
    else:
        print("Usage: python quick_test.py path/to/image.jpg")
        print("\nOr enter image path now:")
        path = input("Image path: ").strip()
        if path and os.path.exists(path):
            predict_flower(path)
        else:
            print("Invalid path!")