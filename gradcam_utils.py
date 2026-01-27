"""
Grad-CAM utilities for visualizing model attention.
"""
import numpy as np
import tensorflow as tf
import cv2


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap for an image.
    
    Args:
        img_array: Preprocessed image array (batch dimension included)
        model: Keras model
        last_conv_layer_name: Name of the last convolutional layer
        pred_index: Index of the predicted class (None = use top prediction)
    
    Returns:
        Heatmap array (normalized to [0, 1])
    """
    # Try to get the layer directly first
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        # Layer found directly, create model with this layer's output
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [last_conv_layer.output, model.output]
        )
    except:
        # If not found, try to find it in a nested base model
        base_model_names = ['vgg16', 'resnet50']
        last_conv_layer = None
        base_model_layer = None
        
        for base_name in base_model_names:
            try:
                base_model_layer = model.get_layer(base_name)
                last_conv_layer = base_model_layer.get_layer(last_conv_layer_name)
                break
            except:
                continue
        
        if last_conv_layer is None:
            raise ValueError(f"Layer '{last_conv_layer_name}' not found")
        
        # For nested models: Use base model's final output as proxy
        # This works around the graph disconnection issue
        
        main_input = model.input
        
        # Find GlobalAveragePooling2D layer which receives base model output
        gap_layer = None
        for layer in model.layers:
            if 'global_average_pooling' in layer.name.lower():
                gap_layer = layer
                break
        
        if gap_layer is None:
            raise ValueError("Cannot find GlobalAveragePooling2D layer")
        
        # Try to access gap_layer.input (base model output)
        # This might be disconnected, so we'll handle the error
        try:
            base_output = gap_layer.input
            # Test if we can create a model with it
            test_model = tf.keras.models.Model([model.inputs], [base_output])
            # If successful, use it
            grad_model = tf.keras.models.Model(
                [model.inputs],
                [base_output, model.output]
            )
        except Exception as e:
            # Graph disconnection - this is expected for saved nested models
            # Raise error so app can use fallback visualization
            raise ValueError(
                f"Graph disconnection: Cannot extract base model output. "
                f"This is a known limitation with saved models containing nested base models. "
                f"The app will use an attention visualization instead. "
                f"Original error: {str(e)[:100]}"
            )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        intermediate_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # Gradient of the predicted class with respect to the intermediate output
    grads = tape.gradient(class_channel, intermediate_output)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by corresponding gradients
    intermediate_output = intermediate_output[0]
    heatmap = intermediate_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    
    # Resize heatmap to match image size (224x224)
    heatmap_np = heatmap.numpy()
    if heatmap_np.shape != (224, 224):
        heatmap_np = cv2.resize(heatmap_np, (224, 224))
    
    return heatmap_np


def overlay_gradcam(img_array_original, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        img_array_original: Original image array (224, 224, 3) in [0, 1]
        heatmap: Heatmap array (224, 224) in [0, 1]
        alpha: Transparency factor for overlay
    
    Returns:
        Overlaid image array (224, 224, 3) in [0, 1]
    """
    # Ensure heatmap is 2D and properly sized
    if len(heatmap.shape) != 2:
        heatmap = heatmap.squeeze()
    if heatmap.shape != (224, 224):
        heatmap = cv2.resize(heatmap, (224, 224))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # Ensure image is in [0, 1] range
    if img_array_original.max() > 1.0:
        img_array_original = img_array_original / 255.0
    
    # Ensure image is 224x224
    if img_array_original.shape[:2] != (224, 224):
        img_array_original = cv2.resize(img_array_original, (224, 224))
    
    # Overlay
    superimposed_img = heatmap * alpha + img_array_original * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 1)
    
    return superimposed_img
