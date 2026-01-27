"""
Evaluate trained models and generate metrics, confusion matrices, and ROC curves.
"""
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from data_utils import get_datasets, CLASS_NAMES

# Create figs directory if it doesn't exist
os.makedirs("figs", exist_ok=True)

def evaluate_model(model_path, model_name, test_ds):
    """
    Evaluate a model and generate metrics, confusion matrix, and ROC curve.
    
    Args:
        model_path: Path to saved model .h5 file
        model_name: Name of model (for file naming)
        test_ds: Test dataset
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found. Skipping evaluation.")
        return
    
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    # Get true labels and predictions
    print("Generating predictions...")
    y_true = []
    y_pred_proba = []
    
    for images, labels in test_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_proba.extend(model.predict(images, verbose=0))
    
    y_true = np.array(y_true)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_pred_proba = np.array(y_pred_proba)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Store metrics in dictionary
    metrics = {
        'accuracy': float(accuracy),
        'macro_avg': {
            'precision': float(precision_macro),
            'recall': float(recall_macro),
            'f1_score': float(f1_macro)
        },
        'weighted_avg': {
            'precision': float(precision_weighted),
            'recall': float(recall_weighted),
            'f1_score': float(f1_weighted)
        },
        'per_class': {}
    }
    
    for i, class_name in enumerate(CLASS_NAMES):
        metrics['per_class'][class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create enhanced confusion matrix with percentages
    plt.figure(figsize=(10, 8))
    cm_percent = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8) * 100
    
    # Create annotations with both counts and percentages
    annotations = []
    for i in range(len(CLASS_NAMES)):
        row = []
        for j in range(len(CLASS_NAMES)):
            row.append(f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)')
        annotations.append(row)
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
    plt.title(f'{model_name} - Confusion Matrix\nAccuracy: {accuracy:.2%}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'figs/{model_name.lower()}_cm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to figs/{model_name.lower()}_cm.png")
    
    # ROC Curve (malignant vs others)
    # Binarize: malignant (class 2) vs (normal + benign)
    y_true_binary = (y_true == 2).astype(int)
    y_pred_proba_binary = y_pred_proba[:, 2]  # Probability of malignant
    
    fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba_binary)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve (Malignant vs Others)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'figs/{model_name.lower()}_roc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to figs/{model_name.lower()}_roc.png")
    print(f"AUC: {roc_auc:.4f}")
    
    # Add AUC to metrics
    metrics['auc'] = float(roc_auc)
    
    # Save metrics to JSON file
    metrics_path = f'figs/{model_name.lower()}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


def main():
    """Evaluate both models."""
    print("Loading test dataset...")
    _, _, test_ds, _ = get_datasets(batch_size=16)
    
    # Evaluate VGG16
    evaluate_model('models/vgg16_final.h5', 'VGG16', test_ds)
    
    # Evaluate ResNet50
    evaluate_model('models/resnet50_final.h5', 'ResNet50', test_ds)
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)


if __name__ == "__main__":
    main()

