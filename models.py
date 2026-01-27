"""
Model architectures for breast ultrasound classification.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications


def build_vgg16_model(num_classes=3):
    """
    Build VGG16-based model for classification.
    
    Args:
        num_classes: Number of output classes (default: 3 for normal/benign/malignant)
    
    Returns:
        Compiled Keras model
    """
    # Base VGG16 (frozen initially)
    base_model = applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model with intermediate output accessible for Grad-CAM
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    # Store base model output for Grad-CAM access
    base_output = x
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model with both outputs (for Grad-CAM access)
    model = keras.Model(inputs, outputs)
    # Store base_output as an attribute for Grad-CAM
    model._base_output = base_output
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model


def build_resnet50_model(num_classes=3):
    """
    Build ResNet50-based model for classification.
    
    Args:
        num_classes: Number of output classes (default: 3 for normal/benign/malignant)
    
    Returns:
        Compiled Keras model
    """
    # Base ResNet50 (frozen initially)
    base_model = applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model with intermediate output accessible for Grad-CAM
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    # Store base model output for Grad-CAM access
    base_output = x
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model with both outputs (for Grad-CAM access)
    model = keras.Model(inputs, outputs)
    # Store base_output as an attribute for Grad-CAM
    model._base_output = base_output
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

