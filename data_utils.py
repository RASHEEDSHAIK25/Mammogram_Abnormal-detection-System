"""
Data loading and preprocessing utilities for breast ultrasound classification.
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path

CLASS_NAMES = ["normal", "benign", "malignant"]

# Dataset paths
NORMAL_PATH = r"D:\Mammogram_Abnormal detection\Dataset_BUSI_with_GT\normal"
BENIGN_PATH = r"D:\Mammogram_Abnormal detection\Dataset_BUSI_with_GT\benign"
MALIGNANT_PATH = r"D:\Mammogram_Abnormal detection\Dataset_BUSI_with_GT\malignant"


def load_image_paths_and_labels():
    """
    Load all image paths from the three class folders and map to labels.
    
    Returns:
        paths: List of image file paths
        labels: List of integer labels (0=normal, 1=benign, 2=malignant)
    """
    paths = []
    labels = []
    
    # Map class names to paths
    class_paths = {
        0: NORMAL_PATH,      # normal
        1: BENIGN_PATH,      # benign
        2: MALIGNANT_PATH    # malignant
    }
    
    for label, folder_path in class_paths.items():
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Dataset folder not found: {folder_path}")
        
        # Get all image files (png, jpg, jpeg)
        image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        for ext in image_extensions:
            for img_path in Path(folder_path).glob(f"*{ext}"):
                # Skip mask files (usually contain 'mask' in filename)
                if 'mask' not in img_path.name.lower():
                    paths.append(str(img_path))
                    labels.append(label)
    
    return paths, labels


def train_val_test_split(paths, labels, test_size=0.15, val_size=0.15, random_state=42):
    """
    Stratified split into train/val/test sets.
    
    Args:
        paths: List of image paths
        labels: List of integer labels
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test)
        random_state: Random seed
    
    Returns:
        (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)
    """
    # First split: train+val vs test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    
    # Second split: train vs val
    # Adjust val_size to be relative to train+val set
    val_size_adjusted = val_size / (1 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, 
        test_size=val_size_adjusted, stratify=train_val_labels, 
        random_state=random_state
    )
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def preprocess_image(image_path, augment=False):
    """
    Preprocess a single image: read, convert to grayscale->RGB, resize, normalize.
    
    Args:
        image_path: Path to image file
        augment: Whether to apply data augmentation
    
    Returns:
        Preprocessed image tensor
    """
    # Read image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    
    # Convert to grayscale then back to RGB (3 channels)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    
    # Data augmentation (before resize for better results)
    if augment:
        # Resize to slightly larger for random crop
        image = tf.image.resize(image, [256, 256])
        # Random crop to 224x224
        image = tf.image.random_crop(image, size=[224, 224, 3])
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        # Random vertical flip
        image = tf.image.random_flip_up_down(image)
        # Random brightness and contrast
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    else:
        # Resize to (224, 224)
        image = tf.image.resize(image, [224, 224])
    
    # Normalize to [0, 1]
    image = image / 255.0
    
    # Ensure still in [0, 1] range after augmentation
    if augment:
        image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image


def make_tf_dataset(file_paths, labels, batch_size, shuffle=True, augment=False):
    """
    Create a TensorFlow dataset from file paths and labels.
    
    Args:
        file_paths: List of image file paths
        labels: List of integer labels
        batch_size: Batch size
        shuffle: Whether to shuffle
        augment: Whether to apply data augmentation
    
    Returns:
        TensorFlow Dataset
    """
    # Convert to tensors
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    
    # Map paths to images
    image_ds = path_ds.map(
        lambda x: preprocess_image(x, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Combine images and labels
    dataset = tf.data.Dataset.zip((image_ds, label_ds))
    
    # Convert labels to one-hot
    num_classes = len(CLASS_NAMES)
    dataset = dataset.map(lambda img, lbl: (img, tf.one_hot(lbl, num_classes)))
    
    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(file_paths))
    
    # Batch
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def get_datasets(batch_size=16):
    """
    Get train, validation, and test datasets.
    
    Args:
        batch_size: Batch size for datasets
    
    Returns:
        (train_ds, val_ds, test_ds, CLASS_NAMES)
    """
    # Load paths and labels
    paths, labels = load_image_paths_and_labels()
    print(f"Loaded {len(paths)} images")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Split into train/val/test
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = \
        train_val_test_split(paths, labels, test_size=0.15, val_size=0.15, random_state=42)
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Create datasets
    train_ds = make_tf_dataset(train_paths, train_labels, batch_size, shuffle=True, augment=True)
    val_ds = make_tf_dataset(val_paths, val_labels, batch_size, shuffle=False, augment=False)
    test_ds = make_tf_dataset(test_paths, test_labels, batch_size, shuffle=False, augment=False)
    
    return train_ds, val_ds, test_ds, CLASS_NAMES

